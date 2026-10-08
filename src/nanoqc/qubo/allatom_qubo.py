"""All-atom interface QUBO builder (``AllAtomInterfaceQUBOBuilder``).

Complete chi1..chiN rotamer states scored with Amber14 in OpenMM on a fixed
backbone, plus the virtual pruned graph used by the module self-test.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import torch
from torch_geometric.data import Data
from nanoqc.qubo.atomistic_structure import _CHI_ATOMS, _SIDECHAIN_NAMES, _apply_sidechain_chis, _backbone_phi_psi, _chi1_angle, _sidechain_chi_angles, _torsion_angle_degrees, read_atomistic_structure
from nanoqc.qubo.coarse_qubo import InterfaceQUBOBuilder, _combinations
from nanoqc.qubo.ising import ising_roundoff_tolerance, qubo_to_ising, validate_qubo_ising_equivalence
from nanoqc.qubo.qubo_types import AA_INDEX, QUBOResult, RotamerTemplate, VariableRecord
from nanoqc.qubo.rotamer_library import _THREE_LETTER, _dunbrack_templates_for_site, _expanded_rotamer_templates, _load_rotamer_bins, _nearest_dunbrack_bin, rotamer_source_metadata
from nanoqc.structure.physical_quality import (StructureQualityError, topology_geometry_audit,
    relaxation_force_audit, RELAX_FORCE_TOLERANCE_KJ_MOL_NM,
    EXTREME_NONBONDED_FLOOR_ANGSTROM)


# A46: L-BFGS (Liu & Nocedal 1989) with a per-atom step cap. Amber ff14SB
# hydroxyl hydrogens (type protein-HO: Ser HG, Thr HG1, Tyr HH, ASH HD2, GLH
# HE2, HYP HD1) have zero Lennard-Jones repulsion, so their Coulomb attraction
# to an oppositely charged atom is unbounded as the distance goes to zero.
# SciPy's L-BFGS-B takes a first trial step of length 1 in coordinate units
# (stp = min(1/dnorm, stpmx) at iteration 0; 1 nm here) and later full
# quasi-Newton steps, and its line search accepts any energy decrease, so it
# can jump into that basin. Capping every iteration's displacement of each
# atom, as GROMACS steepest descent caps it with emstep, keeps each step
# local; the quasi-Newton history is kept, so total travel is limited only by
# the iteration cap. A step cap alone does not remove the basin (7WKI, A50).
MINIMIZER_MAX_ATOM_STEP_NM = 0.03
MINIMIZER_HISTORY = 10
_ARMIJO = 1e-4
_MAX_BACKTRACKS = 40
# OpenMM's CUDA platform accumulates forces only in a 64-bit fixed-point
# buffer, converting by multiplying by 2**32 (OpenMM developer guide; OpenCL
# uses such a buffer too), so no force component above 2**31 kJ/mol/nm is
# representable in any precision mode. The guide does not define what happens
# beyond that range; on the server, three atoms reported the identical
# magnitude 2**31*sqrt(3) (7NXX seed 43), so such output cannot be trusted.
# A carbon 0.5 A from a hydrogen exceeds the limit. Whether that can happen is
# therefore decided from geometry, never from the GPU's own output: if any interacting pair with a movable
# atom is closer than the distance at which the strongest Lennard-Jones pair
# of the System reaches 1/64 of the limit, energy and forces are recomputed on
# the double-precision Reference platform from the same System. Otherwise the
# GPU result is used unchanged.
FIXED_POINT_FORCE_LIMIT_KJ_MOL_NM = 2.0 ** 31
_PAIR_FORCE_BUDGET = FIXED_POINT_FORCE_LIMIT_KJ_MOL_NM / 64


def _nonbonded_exclusions_and_wall(system: Any) -> tuple[set[tuple[int, int]], float] | None:
    """Fully excluded pairs (1-2, 1-3) and the strongest r^-12 wall of a System.

    Returns None when the System has no single NonbondedForce. The wall is
    epsilon*sigma^12 maximized over Lorentz-Berthelot combinations of the
    particle types and over the explicit parameters of every nonzero exception.
    """
    import openmm as mm
    nonbonded = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    if len(nonbonded) != 1:
        return None
    force = nonbonded[0]
    nm, kj = mm.unit.nanometer, mm.unit.kilojoule_per_mole
    types = {(force.getParticleParameters(i)[1].value_in_unit(nm),
              force.getParticleParameters(i)[2].value_in_unit(kj))
             for i in range(force.getNumParticles())}
    strength = max((math.sqrt(e1 * e2) * ((s1 + s2) / 2) ** 12
                    for s1, e1 in types for s2, e2 in types), default=0.0)
    excluded: set[tuple[int, int]] = set()
    for index in range(force.getNumExceptions()):
        i, j, charge, s, e = force.getExceptionParameters(index)
        epsilon = e.value_in_unit(kj)
        if charge.value_in_unit(mm.unit.elementary_charge ** 2) == 0.0 and epsilon == 0.0:
            excluded.add((min(i, j), max(i, j)))
        else:
            strength = max(strength, epsilon * s.value_in_unit(nm) ** 12)
    return excluded, strength


class _ClosePairs:
    """Interacting pairs with a movable atom closer than a radius.

    Frozen atoms never move during a minimization, so their tree is built once;
    each query only places the movable atoms.
    """

    def __init__(self, start_nm: np.ndarray, movable: Sequence[int],
                 excluded: set[tuple[int, int]]) -> None:
        from scipy.spatial import cKDTree
        self._tree_type = cKDTree
        self.movable = np.asarray(sorted(movable), dtype=np.int64)
        moving = set(self.movable.tolist())
        self.frozen = np.asarray([i for i in range(len(start_nm)) if i not in moving], dtype=np.int64)
        self.frozen_tree = cKDTree(start_nm[self.frozen]) if len(self.frozen) else None
        self.excluded = excluded

    def __call__(self, positions_nm: np.ndarray, radius_nm: float) -> list[tuple[int, int]]:
        pairs = set()
        moving_xyz = positions_nm[self.movable]
        if self.frozen_tree is not None:
            for i, neighbours in zip(self.movable, self.frozen_tree.query_ball_point(moving_xyz, radius_nm)):
                for k in neighbours:
                    pair = (min(i, int(self.frozen[k])), max(i, int(self.frozen[k])))
                    if pair not in self.excluded:
                        pairs.add(pair)
        for a, b in self._tree_type(moving_xyz).query_pairs(radius_nm):
            pair = (min(self.movable[a], self.movable[b]), max(self.movable[a], self.movable[b]))
            if pair not in self.excluded:
                pairs.add((int(pair[0]), int(pair[1])))
        return sorted(pairs)


class _ExactForces:
    """Energy and forces at given positions, exact even under severe overlaps."""

    def __init__(self, context: Any, movable: set[int]) -> None:
        import openmm as mm
        self._mm = mm
        self.context = context
        self.movable = np.asarray(sorted(movable), dtype=np.int64)
        self.reference = None
        self.reference_evaluations = 0
        self.guarded = context.getPlatform().getName() not in ("Reference", "CPU")
        self.safe_distance_nm = None
        self.excluded: set[tuple[int, int]] = set()
        self._pairs = None
        if not self.guarded:
            return
        found = _nonbonded_exclusions_and_wall(context.getSystem())
        if found is None:
            self.safe_distance_nm = float("inf")   # unknown pair potential: always exact
            return
        self.excluded, strength = found
        # |F_LJ(r)| <= 48 epsilon sigma^12 / r^13 for every interacting pair.
        self.safe_distance_nm = float((48.0 * strength / _PAIR_FORCE_BUDGET) ** (1 / 13))

    def needs_reference(self, positions_nm: np.ndarray) -> bool:
        if not self.guarded:
            return False
        if not np.isfinite(self.safe_distance_nm):
            return True
        if self._pairs is None:
            # Frozen atoms keep these positions for the life of this object.
            self._pairs = _ClosePairs(positions_nm, self.movable, self.excluded)
        return bool(self._pairs(positions_nm, self.safe_distance_nm))

    def __call__(self, positions_nm: np.ndarray, unit: Any) -> tuple[float, np.ndarray]:
        target = self.context
        if self.needs_reference(positions_nm):
            if self.reference is None:
                mm = self._mm
                self.reference = mm.Context(self.context.getSystem(), mm.VerletIntegrator(0.001),
                                            mm.Platform.getPlatformByName("Reference"))
            target = self.reference
            self.reference_evaluations += 1
        target.setPositions(positions_nm * unit.nanometer)
        state = target.getState(getEnergy=True, getForces=True)
        return (float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)),
                np.asarray(state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoules_per_mole / unit.nanometer), dtype=np.float64))


def _capped_lbfgs(objective, x0: np.ndarray, *, max_iterations: int, max_atom_step: float,
                  force_tolerance: float, history: int = MINIMIZER_HISTORY,
                  callback=None, admissible=None) -> dict[str, Any]:
    """Minimize ``objective`` (energy, gradient) over flattened [atoms, 3] coordinates.

    Every accepted iteration moves each atom by at most ``max_atom_step``.
    Convergence is the exact force criterion only (gradient RMS at most
    ``force_tolerance``); there is no relative-energy stop. A backtracking
    failure with quasi-Newton history retries once along the steepest descent
    before stopping. ``admissible(trial, current)``, if given, can veto a
    trial point; a vetoed step is backtracked like one without sufficient
    decrease.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    energy, gradient = objective(x)
    evaluations, iterations, resets = 1, 0, 0
    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    stop = None
    while True:
        if float(np.sqrt(np.mean(gradient ** 2))) <= force_tolerance:
            stop = "force_tolerance"
            break
        if iterations >= max_iterations:
            stop = "iteration_cap"
            break
        direction = -gradient.copy()
        if pairs:
            alphas = []
            for s, y, rho in reversed(pairs):
                alpha = rho * float(s @ direction)
                alphas.append(alpha)
                direction -= alpha * y
            s, y, _ = pairs[-1]
            direction *= float(s @ y) / float(y @ y)
            for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
                direction += (alpha - rho * float(y @ direction)) * s
            if float(direction @ gradient) >= 0.0:
                pairs.clear()
                resets += 1
                direction = -gradient.copy()
        step = float(np.linalg.norm(direction.reshape(-1, 3), axis=1).max())
        if not pairs or step > max_atom_step:
            direction *= max_atom_step / step
        slope = float(direction @ gradient)
        scale = 1.0
        for _ in range(_MAX_BACKTRACKS):
            trial = x + scale * direction
            if admissible is not None and not admissible(trial, x):
                scale *= 0.5
                continue
            trial_energy, trial_gradient = objective(trial)
            evaluations += 1
            if trial_energy <= energy + _ARMIJO * scale * slope:
                break
            scale *= 0.5
        else:
            if pairs:
                pairs.clear()
                resets += 1
                continue
            stop = "line_search_failed"
            break
        s, y = trial - x, trial_gradient - gradient
        curvature = float(s @ y)
        if curvature > 1e-12 * float(np.linalg.norm(s) * np.linalg.norm(y)):
            pairs.append((s, y, 1.0 / curvature))
            del pairs[:-history]
        x, energy, gradient = trial, trial_energy, trial_gradient
        iterations += 1
        if callback is not None:
            callback(x, s)
    return dict(x=x, energy=energy, gradient=gradient, iterations=iterations,
                evaluations=evaluations, stop=stop, history_resets=resets)


def _minimize_movable_positions(context: Any, positions: np.ndarray,
                                movable: set[int], max_iterations: int,
                                unit: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize the exact OpenMM potential over movable coordinates only.

    OpenMM's LocalEnergyMinimizer can stop with substantial residual force when
    most particles have zero mass to enforce a fixed backbone.  Explicitly
    optimizing only the declared movable coordinates leaves the frozen atoms
    bitwise fixed and permits an independent force-based acceptance audit.
    """
    if max_iterations <= 0 or not movable:
        raise ValueError("Movable minimization needs a positive cap and atom set")
    start = np.asarray(positions, dtype=np.float64)
    if start.ndim != 2 or start.shape[1] != 3 or not np.isfinite(start).all():
        raise ValueError("Movable minimization requires finite [atoms,3] positions")
    indices = np.asarray(sorted(movable), dtype=np.int64)
    if indices[0] < 0 or indices[-1] >= len(start):
        raise ValueError("Movable atom index outside topology")

    exact = _ExactForces(context, movable)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        current = start.copy()
        current[indices] = flat.reshape(-1, 3)
        energy, forces = exact(current, unit)
        gradient = -forces[indices].ravel()
        if not math.isfinite(energy) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite energy or force during movable minimization")
        return energy, gradient

    # Relative objective convergence is unsafe for a severely clashing
    # structure: an enormous potential can change by less than its relative
    # tolerance while the remaining forces are still enormous, so only the
    # force criterion ends the minimization early.
    # Amber polar hydrogens with zero Lennard-Jones (HO) are pulled without
    # bound onto an oppositely charged atom once they overlap it, so no step
    # may bring an interacting pair below the absolute near-coincidence floor
    # closer than it already is. A physical minimum never has such a pair.
    found = _nonbonded_exclusions_and_wall(context.getSystem())
    floor_nm = EXTREME_NONBONDED_FLOOR_ANGSTROM / 10.0
    vetoes = [0]

    close_pairs = _ClosePairs(start, indices, found[0]) if found is not None else None

    def admissible(trial: np.ndarray, current: np.ndarray) -> bool:
        if close_pairs is None:
            return True
        trial_xyz = start.copy()
        trial_xyz[indices] = trial.reshape(-1, 3)
        current_xyz = start.copy()
        current_xyz[indices] = current.reshape(-1, 3)
        for i, j in close_pairs(trial_xyz, floor_nm):
            if (np.linalg.norm(trial_xyz[i] - trial_xyz[j])
                    < np.linalg.norm(current_xyz[i] - current_xyz[j])):
                vetoes[0] += 1
                return False
        return True

    result = _capped_lbfgs(objective, start[indices].ravel(), max_iterations=max_iterations,
                           max_atom_step=MINIMIZER_MAX_ATOM_STEP_NM,
                           force_tolerance=RELAX_FORCE_TOLERANCE_KJ_MOL_NM,
                           admissible=admissible)
    final = start.copy()
    final[indices] = result["x"].reshape(-1, 3)
    if not np.isfinite(final).all():
        raise FloatingPointError("Nonfinite coordinates after movable minimization")
    context.setPositions(final * unit.nanometer)
    converged = result["stop"] == "force_tolerance"
    return final, dict(minimizer="lbfgs_exact_movable_capped_atom_step",
                       minimizer_max_atom_step_nm=MINIMIZER_MAX_ATOM_STEP_NM,
                       minimizer_history=MINIMIZER_HISTORY,
                       minimizer_iterations=result["iterations"],
                       minimizer_evaluations=result["evaluations"],
                       minimizer_stop_reason=result["stop"],
                       minimizer_reported_success=converged,
                       minimizer_force_verified_success=converged,
                       minimizer_restart_count=result["history_resets"],
                       minimizer_reference_platform_evaluations=exact.reference_evaluations,
                       minimizer_reference_distance_nm=exact.safe_distance_nm,
                       minimizer_overlap_floor_vetoes=vetoes[0])


# A47 (revised): a discrete rotamer state is geometrically impossible if it
# puts an interacting atom pair under either A45 input floor: any two atoms
# closer than the 0.4-A near-coincidence floor, or two heavy atoms closer
# than 1.0 A, against a fixed atom or against a state at another site. Their
# raw r^-12 energies (up to 1e14 kcal/mol) would exhaust the float64
# precision of the QUBO. Hydrogen contacts between 0.4 and 1.0 A are not
# impossible (relaxation removes them) and keep their Amber energies.
DISCRETE_STATE_CONTACT_FLOOR_ANGSTROM = 1.0          # heavy-heavy
DISCRETE_STATE_ALL_ATOM_FLOOR_ANGSTROM = EXTREME_NONBONDED_FLOOR_ANGSTROM   # any elements


# A50: Amber ff14SB (type inherited from Cornell et al. 1995) gives hydroxyl
# hydrogens (type protein-HO: Ser HG, Thr HG1, Tyr HH, ASH HD2, GLH HE2, HYP
# HD1; the only zero-epsilon protein type) no Lennard-Jones term. Their parent oxygen normally shields them, but
# not from atoms the oxygen is 1-3 excluded from: Tyr HH and its own ring
# carbon CE1/CE2 form a 1-4 pair with an attractive Coulomb term and no
# repulsion, so a relaxation can slide HH onto the ring (7WKI seed 42). These
# hydrogens get CHARMM's polar-hydrogen parameters (type H: epsilon 0.046
# kcal/mol, Rmin/2 0.2245 A; MacKerell et al. 1998, kept in CHARMM36), and
# their 1-4 pairs get the force field's own 1-4 Lennard-Jones scaling (0.5).
# At hydrogen-bond distances the added term is at most ~0.1 kcal/mol; at 1 A
# from a ring carbon it is about +270 kcal/mol. This hybrid is a
# study-specific modification, not a published validated force field.
POLAR_HYDROGEN_SIGMA_NM = 2 * 0.02245 / 2 ** (1 / 6)
POLAR_HYDROGEN_EPSILON_KJ_MOL = 0.046 * 4.184


def _shield_zero_lj_polar_hydrogens(system: Any, topology: Any) -> dict[str, Any]:
    """Give zero-LJ hydroxyl hydrogens a small repulsive core (in place)."""
    import openmm as mm
    nonbonded = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    if len(nonbonded) != 1:
        raise ValueError("Polar-hydrogen shielding requires exactly one NonbondedForce")
    force = nonbonded[0]
    nm, kj, e2 = mm.unit.nanometer, mm.unit.kilojoule_per_mole, mm.unit.elementary_charge ** 2
    atoms = list(topology.atoms())
    oxygen_bonded = set()
    for a, b in topology.bonds():
        for h, o in ((a, b), (b, a)):
            if (h.element is not None and h.element.symbol.upper() == "H"
                    and o.element is not None and o.element.symbol.upper() == "O"):
                oxygen_bonded.add(h.index)
    shielded = []
    for index in sorted(oxygen_bonded):
        charge, sigma, epsilon = force.getParticleParameters(index)
        if epsilon.value_in_unit(kj) == 0.0 and charge.value_in_unit(mm.unit.elementary_charge) != 0.0:
            force.setParticleParameters(index, charge, POLAR_HYDROGEN_SIGMA_NM * nm,
                                        POLAR_HYDROGEN_EPSILON_KJ_MOL * kj)
            shielded.append(index)
    if not shielded:
        return dict(shielded_hydrogens=0, one_four_pairs_updated=0)
    # The force field's 1-4 Lennard-Jones scale, read from its own exceptions.
    scales = []
    for k in range(force.getNumExceptions()):
        i, j, q, s, e = force.getExceptionParameters(k)
        e_ij = e.value_in_unit(kj)
        if e_ij > 0.0:
            ei = force.getParticleParameters(i)[2].value_in_unit(kj)
            ej = force.getParticleParameters(j)[2].value_in_unit(kj)
            if ei > 0 and ej > 0 and i not in shielded and j not in shielded:
                scales.append(e_ij / math.sqrt(ei * ej))
    if not scales or max(scales) - min(scales) > 1e-6 * max(scales):
        raise ValueError("Inconsistent 1-4 Lennard-Jones scaling in the force field")
    scale = float(np.mean(scales))
    shielded_set = set(shielded)
    updated = 0
    for k in range(force.getNumExceptions()):
        i, j, q, s, e = force.getExceptionParameters(k)
        if (i in shielded_set or j in shielded_set) and q.value_in_unit(e2) != 0.0:
            _, si, ei = force.getParticleParameters(i)
            _, sj, ej = force.getParticleParameters(j)
            force.setExceptionParameters(
                k, i, j, q, 0.5 * (si.value_in_unit(nm) + sj.value_in_unit(nm)) * nm,
                scale * math.sqrt(ei.value_in_unit(kj) * ej.value_in_unit(kj)) * kj)
            updated += 1
    return dict(shielded_hydrogens=len(shielded), one_four_pairs_updated=updated,
                one_four_lj_scale=scale, sigma_nm=POLAR_HYDROGEN_SIGMA_NM,
                epsilon_kj_mol=POLAR_HYDROGEN_EPSILON_KJ_MOL,
                residues=sorted({f"{atoms[i].residue.chain.id}:{atoms[i].residue.id}:"
                                 f"{atoms[i].residue.name}" for i in shielded}))


def selection_is_geometry_admissible(metadata: Mapping[str, Any], bits: Sequence[int]) -> bool:
    """Whether a one-hot bit string avoids every A47 forbidden state and pair.

    For such a selection the QUBO energy equals the Amber energy (vacuum);
    a forbidden selection carries the penalty instead and cannot be compared.
    """
    chosen = {i for i, bit in enumerate(bits) if int(bit)}
    if chosen & set(int(v) for v in metadata.get("forbidden_variables", ())):
        return False
    return not any(int(i) in chosen and int(j) in chosen
                   for i, j in metadata.get("forbidden_variable_pairs", ()))


def _heavy_mask(topology: Any) -> np.ndarray:
    return np.asarray([atom.element is not None and atom.element.symbol.upper() != "H"
                       for atom in topology.atoms()], dtype=bool)


def _impossible_contact(distance_angstrom: float, heavy_pair: bool) -> bool:
    return (distance_angstrom < DISCRETE_STATE_ALL_ATOM_FLOOR_ANGSTROM
            or (heavy_pair and distance_angstrom < DISCRETE_STATE_CONTACT_FLOOR_ANGSTROM))


_ASSIGNMENT_SEARCH_NODE_LIMIT = 1_000_000


def _discrete_state_admissibility(builder: Any) -> dict[str, Any]:
    """Which candidate states and cross-site state pairs are geometrically possible."""
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    candidates = builder.candidates
    count = len(candidates)
    atoms = list(builder.topology.atoms())
    label = lambda i: (f"{atoms[i].residue.chain.id}:{atoms[i].residue.id}:"
                       f"{atoms[i].residue.name}:{atoms[i].name}")
    found = _nonbonded_exclusions_and_wall(builder.system)
    excluded = found[0] if found is not None else set()
    heavy = _heavy_mask(builder.topology)
    floor_nm = DISCRETE_STATE_CONTACT_FLOOR_ANGSTROM / 10.0
    side_chain = set().union(*(set(int(i) for i in c["indices"]) for c in candidates))
    fixed = np.array(sorted(set(range(len(builder.base_positions))) - side_chain), dtype=np.int64)
    tree = cKDTree(builder.base_positions[fixed])
    single_ok = np.ones(count, dtype=bool)
    forbidden_singles = []
    for v, candidate in enumerate(candidates):
        closest = None
        for atom, xyz in zip(candidate["indices"], candidate["positions"]):
            for k in tree.query_ball_point(xyz, floor_nm):
                other = int(fixed[k])
                if (min(atom, other), max(atom, other)) in excluded:
                    continue
                distance = float(np.linalg.norm(xyz - builder.base_positions[other])) * 10.0
                if not _impossible_contact(distance, bool(heavy[atom] and heavy[other])):
                    continue
                if closest is None or distance < closest[0]:
                    closest = (distance, int(atom), other)
        if closest is not None:
            single_ok[v] = False
            forbidden_singles.append(dict(variable=v, residue_id=candidate["residue_id"],
                                          distance_angstrom=closest[0],
                                          atoms=[label(closest[1]), label(closest[2])]))
    pair_ok = np.ones((count, count), dtype=bool)
    forbidden_pairs = []
    for i in range(count):
        for j in range(i + 1, count):
            left, right = candidates[i], candidates[j]
            if left["site"] == right["site"]:
                continue
            distances = cdist(left["positions"], right["positions"]) * 10.0
            for a, b in zip(*np.nonzero(distances < DISCRETE_STATE_CONTACT_FLOOR_ANGSTROM)):
                x, y = int(left["indices"][a]), int(right["indices"][b])
                if ((min(x, y), max(x, y)) not in excluded
                        and _impossible_contact(float(distances[a, b]), bool(heavy[x] and heavy[y]))):
                    pair_ok[i, j] = pair_ok[j, i] = False
                    forbidden_pairs.append(dict(variables=[i, j],
                                                distance_angstrom=float(distances[a, b]),
                                                atoms=[label(x), label(y)]))
                    break
    return dict(single_ok=single_ok, pair_ok=pair_ok, audit=dict(
        heavy_atom_floor_angstrom=DISCRETE_STATE_CONTACT_FLOOR_ANGSTROM,
        all_atom_floor_angstrom=DISCRETE_STATE_ALL_ATOM_FLOOR_ANGSTROM,
        definition=("interacting atom pairs (1-2 and 1-3 excluded) under 0.4 A, or heavy-atom "
                    "pairs under 1.0 A; unrelaxed discrete states; singles against atoms outside "
                    "every Active side chain"),
        forbidden_single_count=len(forbidden_singles), forbidden_singles=forbidden_singles[:100],
        forbidden_pair_count=len(forbidden_pairs), forbidden_pairs=forbidden_pairs[:100]))


class _SearchLimitExceeded(RuntimeError):
    """The bounded admissible-assignment search gave up before deciding."""


def _admissible_assignment(site_to_variables: Mapping[int, Sequence[int]], single_ok: np.ndarray,
                           pair_ok: np.ndarray, *, order) -> Optional[list[int]]:
    """One state per site with no forbidden single or pair; depth-first in ``order``."""
    sites = list(site_to_variables)
    nodes = [0]

    def extend(chosen: list[int]) -> Optional[list[int]]:
        if len(chosen) == len(sites):
            return chosen
        for v in order(site_to_variables[sites[len(chosen)]]):
            v = int(v)
            nodes[0] += 1
            if nodes[0] > _ASSIGNMENT_SEARCH_NODE_LIMIT:
                raise _SearchLimitExceeded("Admissible-assignment search exceeded its node limit")
            if single_ok[v] and all(pair_ok[v, u] for u in chosen):
                found = extend(chosen + [v])
                if found is not None:
                    return found
        return None

    return extend([])


def _internal_candidate_overlaps(positions: np.ndarray, atoms: Mapping[str, int],
                                 bonds: Mapping[int, set[int]]) -> list[dict[str, Any]]:
    """Find topology-excluded near coincidences within one rotamer residue."""
    names = {index: name for name, index in atoms.items()}
    indices = sorted(names)
    overlaps = []
    for offset, left in enumerate(indices):
        excluded = set(bonds[left])
        for neighbor in bonds[left]:
            excluded.update(bonds[neighbor])
        for right in indices[offset + 1:]:
            if right in excluded:
                continue
            distance = float(10 * np.linalg.norm(positions[left] - positions[right]))
            if distance < EXTREME_NONBONDED_FLOOR_ANGSTROM:
                overlaps.append(dict(distance_angstrom=distance,
                                     atoms=[names[left], names[right]]))
    return overlaps



def _virtual_pruned_graph(site_count: int = 6) -> Data:
    """Create a deterministic interface-like PyG graph for executable tests."""

    if site_count < 5:
        raise ValueError("At least five active sites are required by adaptive pruning")
    amino_acids = "DEKRQNSTYFIL"[:site_count]
    ligand_count = 12
    node_count = site_count + ligand_count
    x = torch.zeros((node_count, 21), dtype=torch.float32)
    for index, aa in enumerate(amino_acids):
        x[index, AA_INDEX[aa]] = 1.0
    for index in range(site_count, node_count):
        x[index, AA_INDEX["A"]] = 1.0
        x[index, -1] = 1.0

    vhh_pos = [
        [3.8 * index, 0.45 * math.sin(index), 0.20 * math.cos(index)]
        for index in range(site_count)
    ]
    ligand_pos = [
        [3.0 * index, 4.0 + 0.25 * math.cos(index), 0.35 * math.sin(index)]
        for index in range(ligand_count)
    ]
    pos = torch.tensor(vhh_pos + ligand_pos, dtype=torch.float32)
    distances = torch.cdist(pos, pos)
    edge_index = ((distances < 8.0) & (distances > 0)).nonzero().t().long()
    data = Data(x=x, pos=pos, edge_index=edge_index)
    data.is_active = torch.tensor([True] * site_count + [False] * ligand_count)
    data.is_frozen_environment = torch.tensor(
        [False] * site_count + [True] * ligand_count
    )
    data.selected_vhh_mask = data.is_active.clone()
    data.interface_score = torch.linspace(1.0, 0.1, node_count)
    data.original_node_index = torch.arange(node_count)
    data.node_chain_id = torch.tensor([0] * site_count + [1] * ligand_count)
    data.residue_ids = [
        *(f"H:{index + 1}" for index in range(site_count)),
        *(f"A:{index + 1}" for index in range(ligand_count)),
    ]
    data.pdb_id = "VIRTUAL"
    data.source_id = "subgraph_to_qubo.py::__main__"
    return data



def _openmm_context(mm: Any, system: Any, integrator: Any) -> Any:
    """Use an explicit recorded platform; never silently fall back on failure."""
    name = os.environ.get("QP_OPENMM_PLATFORM", "Reference")
    if name not in ("Reference", "CPU", "CUDA"):
        raise ValueError("QP_OPENMM_PLATFORM must be Reference, CPU or CUDA")
    properties = {}
    if name == "CPU":
        properties["Threads"] = os.environ.get("OPENMM_CPU_THREADS", "8")
    elif name == "CUDA":
        precision = os.environ.get("QP_OPENMM_PRECISION", "double")
        if precision not in ("single", "mixed", "double"):
            raise ValueError("QP_OPENMM_PRECISION must be single, mixed or double")
        properties = {"Precision": precision, "DeviceIndex": os.environ.get("QP_OPENMM_DEVICE", "0")}
    return mm.Context(system, integrator, mm.Platform.getPlatformByName(name), properties)


class AllAtomInterfaceQUBOBuilder:
    """Amber14 fixed-backbone adaptive multi-state side-chain QUBO.

    Formal all-atom validation uses complete Dunbrack 2010 side-chain rotamer
    states (chi1..chiN) at the residue's backbone phi/psi bin. Chi1 is expanded
    by the configured Dunbrack sigma offsets while distal chi values follow the
    rotamer's statistical means. Amber14 single-candidate energies pre-screen
    the pool; 3--6 states/site are retained under a global <=30-variable budget.
    An explicit chi1_angles sequence remains available only as a legacy
    controlled-ablation override, fed only from an explicit
    ``case["chi1_angles"]`` JSON manifest field -- never auto-populated from
    a coarse ``InterfaceQUBOBuilder`` solve. The coarse builder's rotamer
    angles come from an approximate local frame (nearest same-chain CA
    neighbors plus nearest ligand atom, see ``_local_frame``), not a real
    N-CA-CB-CG dihedral, so they are not physically meaningful chi1 values
    here and must never be passed as this override.

    No native/reference structure is accepted by this builder. Missing heavy atoms and unsupported templates
    fail rather than inventing atoms. The primary protocol uses vacuum NoCutoff; optional GBN2 is a
    pre-declared sensitivity model. These energies are packing/reconstruction proxies, NOT binding free energy.
    """

    def __init__(self, structure_path: Path, active_residues: Sequence[str], *,
                 chi1_angles: Optional[Sequence[float]] = None,  # legacy ablation override only;
                 # real N-CA-CB-CG dihedral degrees -- never a coarse InterfaceQUBOBuilder solve
                 site_scores: Optional[Sequence[float]] = None, seed: int = 42,
                 candidate_relax_iterations: int = 0,
                 rotamer_mode: str = "legacy",
                 rotamer_library_path: Optional[Path] = None,
                 rotamer_probability_floor: float = 1e-4,
                 rotamer_sigma_offsets: Sequence[float] = (-1.0,0.0,1.0),
                 solvent_model: str = "vacuum"):
        import openmm as mm
        from openmm import app, unit
        import random
        import gemmi
        self.mm, self.app, self.unit = mm, app, unit
        if candidate_relax_iterations < 0:
            raise ValueError("Candidate relaxation iterations must be nonnegative")
        self.candidate_relax_iterations = candidate_relax_iterations
        self.rotamer_mode=str(rotamer_mode)
        if self.rotamer_mode not in ("legacy","dunbrack2010","pyrosetta_dun10"):
            raise ValueError("rotamer_mode must be legacy, dunbrack2010 or pyrosetta_dun10")
        self.rotamer_library_path=None if rotamer_library_path is None else Path(rotamer_library_path)
        self.rotamer_probability_floor=float(rotamer_probability_floor)
        self.rotamer_sigma_offsets=tuple(float(v) for v in rotamer_sigma_offsets)
        self.solvent_model=str(solvent_model).lower()
        if self.solvent_model not in ("vacuum","gbn2"):
            raise ValueError("solvent_model must be vacuum or gbn2")
        # Only the vacuum model is exactly pair-decomposable; see build().
        self.pair_decomposition_exact=self.solvent_model=="vacuum"
        if chi1_angles is not None:
            chi1_angles=tuple(float(a) for a in chi1_angles)
            if not 2 <= len(chi1_angles) <= 6 or not np.isfinite(chi1_angles).all():
                raise ValueError("Legacy chi1_angles override requires 2--6 finite angles")
            if len({round(float(a)%360,8) for a in chi1_angles})!=len(chi1_angles):
                raise ValueError("Duplicate chi1 angles modulo 360")
        self.chi1_angles_override=chi1_angles
        ids=list(active_residues)
        if not ids or len(set(ids))!=len(ids):
            raise ValueError("Active residues must be unique and nonempty")
        if site_scores is None:
            self.site_scores=np.zeros(len(ids),dtype=np.float64)
        else:
            self.site_scores=np.asarray(site_scores,dtype=np.float64)
            if self.site_scores.shape!=(len(ids),) or not np.isfinite(self.site_scores).all():
                raise ValueError("site_scores must be finite and aligned with active_residues")
        if chi1_angles is None and len(ids)*3>30:
            raise ValueError("Adaptive all-atom mode requires at most 10 Active residues under the 30-variable budget")
        if chi1_angles is not None and len(ids)*len(chi1_angles)>30:
            raise ValueError("Legacy chi1 angle override exceeds the 30-variable budget")
        structure_path=Path(structure_path)
        structure=gemmi.read_structure(str(structure_path))
        if len(structure)!=1:
            raise ValueError("Prepare a single-model structure before all-atom construction")
        if any(a.altloc not in ("\x00"," ","") for c in structure[0] for r in c for a in r):
            raise ValueError("Resolve alternate conformers before force-field preparation")
        self.source_structure=structure_path
        # Explicitly validate selected canonical heavy atoms before hydrogen addition.
        protein=read_atomistic_structure(structure_path)
        allatom_dunbrack_bins={}
        allatom_backbone_angles={}
        if self.rotamer_mode in ("dunbrack2010","pyrosetta_dun10") and chi1_angles is None:
            requested=set()
            for rid in ids:
                residue_name=protein[rid]["name"]
                aa=gemmi.find_tabulated_residue(residue_name).one_letter_code
                if aa in "AG": continue
                phi,psi=_backbone_phi_psi(protein,rid)
                allatom_backbone_angles[rid]=(phi,psi)
                requested.add((_THREE_LETTER[aa],_nearest_dunbrack_bin(phi),_nearest_dunbrack_bin(psi)))
            allatom_dunbrack_bins=_load_rotamer_bins(self.rotamer_mode,self.rotamer_library_path,requested) if requested else {}
        for rid in ids:
            if rid not in protein or protein[rid]["name"] in ("ALA","GLY","PRO","CYS"):
                raise ValueError(f"Active site has no supported safe acyclic side-chain search: {rid}")
            needed=set(("N","CA","C","O"))|set(_SIDECHAIN_NAMES[protein[rid]["name"]].split())
            if needed-set(protein[rid]["atoms"]):
                raise ValueError(f"Incomplete Active heavy atoms: {rid}")
        suffix=structure_path.name.lower()
        import gzip
        opener=gzip.open if suffix.endswith(".gz") else open
        with opener(structure_path,"rt") as handle:
            parsed=(app.PDBxFile(handle) if suffix.endswith((".cif",".cif.gz")) else app.PDBFile(handle))
        self.input_quality = topology_geometry_audit(parsed.topology,
            np.asarray(parsed.positions.value_in_unit(unit.nanometer)))
        if not self.input_quality["topology_passed"]:
            raise StructureQualityError("Prepared protein contains a peptide-chain break",
                category="input_topology", audit=self.input_quality)
        if self.solvent_model=="vacuum":
            self.forcefield=app.ForceField("amber14-all.xml")
        else:
            # OpenMM's Amber implicit-solvent GBN2 parameters; no explicit
            # solvent particles are added. This is a sensitivity model, not
            # the frozen primary structural protocol.
            self.forcefield=app.ForceField("amber14-all.xml","implicit/gbn2.xml")
        modeller=app.Modeller(parsed.topology,parsed.positions)
        state=random.getstate()
        try:
            random.seed(seed)
            modeller.addHydrogens(self.forcefield, platform=mm.Platform.getPlatformByName("Reference"))
        finally:
            random.setstate(state)
        self.topology=modeller.topology
        self.base_positions=np.asarray(modeller.positions.value_in_unit(unit.nanometer),dtype=float)
        self.preparation_quality=topology_geometry_audit(self.topology,self.base_positions)
        self.system=self.forcefield.createSystem(
            self.topology,nonbondedMethod=app.NoCutoff,
            constraints=None,rigidWater=False,removeCMMotion=False
        )
        self.polar_hydrogen_shielding=_shield_zero_lj_polar_hydrogens(self.system,self.topology)
        self.energy_force_groups={}
        for i, force in enumerate(self.system.getForces()):
            if i>=32: raise ValueError("Energy audit supports at most 32 force groups")
            force.setForceGroup(i)
            self.energy_force_groups[f"{i}:{type(force).__name__}"]=i
        self.integrator=mm.VerletIntegrator(.001)
        self.context=_openmm_context(mm, self.system, self.integrator)
        residues={}
        for residue in self.topology.residues():
            key=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            if key in residues:
                raise ValueError(f"Duplicate topology residue ID: {key}")
            residues[key]=residue
        bonds={i:set() for i in range(self.topology.getNumAtoms())}
        for a,b in self.topology.bonds():
            bonds[a.index].add(b.index); bonds[b.index].add(a.index)
        self.active_residues=ids; self.candidates=[]; self.site_to_variables={}; self.movable=set()
        raw_candidates_by_site: dict[int, list[int]] = {}
        residue_one_letter: dict[int, str] = {}
        raw_pool_sizes: dict[int, int] = {}
        site_atoms: dict[int, dict[str, int]] = {}
        for site,rid in enumerate(ids):
            residue=residues[rid]; atoms={a.name:a.index for a in residue.atoms()}
            site_atoms[site] = atoms
            one_letter=gemmi.find_tabulated_residue(residue.name).one_letter_code
            residue_one_letter[site]=one_letter
            ca,cb=atoms["CA"],atoms["CB"]
            if cb not in bonds[ca]:
                raise ValueError(f"Missing CA-CB bond: {rid}")
            moving={cb}; frontier=[cb]
            while frontier:
                i=frontier.pop()
                for j in bonds[i]:
                    if {i,j}=={ca,cb}: continue
                    if j not in moving: moving.add(j); frontier.append(j)
            if ca in moving or not moving.issubset(set(atoms.values())):
                raise ValueError(f"Cyclic/crosslinked Active side chain unsupported: {rid}")
            indices=np.array(sorted(moving),dtype=int)
            if self.chi1_angles_override is None:
                if self.rotamer_mode in ("dunbrack2010","pyrosetta_dun10") and one_letter not in "AG":
                    phi,psi=allatom_backbone_angles[rid]
                    templates=_dunbrack_templates_for_site(
                        allatom_dunbrack_bins,one_letter,phi,psi,
                        probability_floor=self.rotamer_probability_floor,
                        sigma_offsets=self.rotamer_sigma_offsets,
                        ensure_chi1_wells=(self.rotamer_mode=="pyrosetta_dun10"),
                    )
                else:
                    templates=_expanded_rotamer_templates(one_letter)
            else:
                templates=tuple(RotamerTemplate(float(angle),1.0/len(self.chi1_angles_override),(float(angle),),(),"legacy_override")
                                for angle in self.chi1_angles_override)
            raw_pool_sizes[site]=len(templates)
            variables=[]
            for template in templates:
                targets=(template.chi_degrees if (self.rotamer_mode in ("dunbrack2010","pyrosetta_dun10") and self.chi1_angles_override is None)
                         else (template.chi1_degrees,))
                full=_apply_sidechain_chis(self.base_positions,atoms,bonds,residue.name,targets)
                coordinates=full[indices].copy()
                variables.append(len(self.candidates))
                self.candidates.append(dict(
                    site=site,residue_id=rid,residue_name=residue.name,
                    angle=float(template.chi1_degrees),chi_degrees=tuple(float(v) for v in targets),
                    prior_probability=float(template.prior_probability),
                    indices=indices,positions=coordinates))
            raw_candidates_by_site[site]=variables
            self.movable.update(moving)
        if candidate_relax_iterations:
            for group in raw_candidates_by_site.values():
                moving = set(self.candidates[group[0]]["indices"])
                system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(self.system))
                for i in range(len(self.base_positions)):
                    if i not in moving:
                        system.setParticleMass(i, 0)
                integrator = mm.VerletIntegrator(.001)
                context = _openmm_context(mm, system, integrator)
                for variable in group:
                    positions, minimization = _minimize_movable_positions(
                        context, self.positions_for_variables([variable]), moving,
                        candidate_relax_iterations, unit)
                    fixed = sorted(set(range(len(positions)))-moving)
                    if not np.allclose(positions[fixed], self.base_positions[fixed], atol=1e-10, rtol=0):
                        raise AssertionError("Candidate preparation moved fixed atoms")
                    candidate = self.candidates[variable]
                    candidate["positions"] = positions[candidate["indices"]].copy()
                    candidate["minimization"] = minimization
                del context, integrator

        self.candidate_quality_exclusions=[]
        for site,variables in raw_candidates_by_site.items():
            for variable in variables:
                overlap=_internal_candidate_overlaps(
                    self.positions_for_variables([variable]),site_atoms[site],bonds)
                if overlap:
                    self.candidate_quality_exclusions.append(dict(
                        site=site,residue_id=ids[site],raw_variable=variable,
                        chi_degrees=self.candidates[variable]["chi_degrees"],
                        overlaps=overlap))
                    self.candidates[variable]["invalid_internal_geometry"]=True

        # Adaptive retention after optional raw-candidate relaxation.
        if self.chi1_angles_override is None:
            helper=InterfaceQUBOBuilder(
                min_variables=3*len(ids), max_variables=30, max_sites=len(ids)
            )
            pseudo_nodes=np.arange(len(ids),dtype=np.int64)
            pseudo_aas=[residue_one_letter[i] for i in range(len(ids))]
            counts=helper._allocate_rotamer_counts(
                pseudo_nodes,pseudo_aas,self.site_scores
            )
            retained_old_indices=[]; retained_groups={}
            for site,count in enumerate(counts):
                valid=[variable for variable in raw_candidates_by_site[site]
                       if not self.candidates[variable].get("invalid_internal_geometry",False)]
                if len(valid)<count:
                    raise StructureQualityError(
                        f"Only {len(valid)} internally valid rotamers at {ids[site]}; need {count}",
                        category="candidate_geometry",
                        audit=dict(site=site,residue_id=ids[site],needed=count,
                                   generated=len(raw_candidates_by_site[site]),
                                   excluded=[item for item in self.candidate_quality_exclusions
                                             if item["site"]==site]))
                ranked=sorted(
                    valid,
                    key=lambda variable: (
                        self.energy(self.positions_for_variables([variable])),
                        abs(float(self.candidates[variable]["angle"])),
                        int(variable),
                    ),
                )
                chosen=ranked[:count]
                retained_groups[site]=chosen
                retained_old_indices.extend(chosen)
            old_candidates=self.candidates
            old_to_new={old:new for new,old in enumerate(retained_old_indices)}
            self.candidates=[old_candidates[old] for old in retained_old_indices]
            self.site_to_variables={
                site:tuple(old_to_new[old] for old in retained_groups[site])
                for site in range(len(ids))
            }
            self.raw_rotamer_pool_sizes=[raw_pool_sizes[i] for i in range(len(ids))]
            self.retained_rotamers_per_site=[len(self.site_to_variables[i]) for i in range(len(ids))]
        else:
            if self.candidate_quality_exclusions:
                raise StructureQualityError(
                    "Legacy fixed-angle candidates contain internal atomic overlaps",
                    category="candidate_geometry",
                    audit=dict(excluded=self.candidate_quality_exclusions))
            self.site_to_variables={}
            cursor=0
            for site in range(len(ids)):
                width=len(raw_candidates_by_site[site])
                self.site_to_variables[site]=tuple(range(cursor,cursor+width))
                cursor+=width
            self.raw_rotamer_pool_sizes=[len(self.chi1_angles_override)]*len(ids)
            self.retained_rotamers_per_site=[len(self.chi1_angles_override)]*len(ids)

    def positions_for_variables(self, variables: Sequence[int]) -> np.ndarray:
        """Apply zero or one candidate per site; partial assignments support decomposition."""
        positions=self.base_positions.copy(); used=set()
        for variable in variables:
            c=self.candidates[int(variable)]
            if c["site"] in used: raise ValueError("Multiple candidates for one site")
            used.add(c["site"]); positions[c["indices"]]=c["positions"]
        return positions

    def energy(self, positions: np.ndarray) -> float:
        """Full Amber14 potential including bonded and exception terms, kcal/mol."""
        self.context.setPositions(positions*self.unit.nanometer)
        value=float(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(self.unit.kilocalorie_per_mole))
        if not math.isfinite(value): raise FloatingPointError("Nonfinite all-atom energy")
        return value

    def positions_for_chi_assignment(
        self, assignment: Mapping[str, Sequence[float]]
    ) -> np.ndarray:
        """Apply explicit residue->chi1..chiN targets without candidate projection.

        Used by TRAIN-ONLY coarse-to-Amber calibration so the atomistic target
        is evaluated for the exact same multi-chi rotamer represented by the
        coarse candidate metadata.
        """
        positions=self.base_positions.copy()
        residue_lookup={}
        bond_graph={}
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            residue_lookup[rid]=residue
        for bond in self.topology.bonds():
            a,b=bond[0].index,bond[1].index
            bond_graph.setdefault(a,set()).add(b);bond_graph.setdefault(b,set()).add(a)
        for rid,targets in assignment.items():
            if rid not in self.active_residues:
                raise ValueError(f"Calibration assignment contains non-active residue: {rid}")
            residue=residue_lookup[rid]
            atoms={a.name:a.index for a in residue.atoms()}
            target_tuple=tuple(float(v) for v in targets)
            expected=len(_CHI_ATOMS.get(residue.name, ()))
            if expected == 0 or len(target_tuple) != expected:
                raise ValueError(
                    f"{rid} expects {expected} chi angles, received {len(target_tuple)}"
                )
            positions=_apply_sidechain_chis(
                positions, atoms, bond_graph, residue.name, target_tuple
            )
            observed=_sidechain_chi_angles(
                {name:positions[index] for name,index in atoms.items()}, residue.name
            )
            if len(observed)!=len(target_tuple):
                raise AssertionError("Explicit multi-chi assignment dimensionality mismatch")
            for got,want in zip(observed,target_tuple):
                if abs((float(got)-float(want)+180)%360-180)>1e-4:
                    raise AssertionError(
                        f"Explicit multi-chi calibration rotation mismatch for {rid}: "
                        f"observed={observed}, target={target_tuple}"
                    )
        return positions

    def chi_assignment_contact(
        self, assignment: Mapping[str, Sequence[float]]
    ) -> Optional[dict[str, Any]]:
        """Closest interacting contact under the A47 floors for a chi assignment, or None.

        Pairs involve at least one Active side-chain atom; 1-2 and 1-3 pairs are
        excluded. A non-None result is a geometrically impossible state.
        """
        if getattr(self, "_contact_pairs", None) is None:
            found = _nonbonded_exclusions_and_wall(self.system)
            self._contact_pairs = _ClosePairs(self.base_positions, sorted(self.movable),
                                              found[0] if found is not None else set())
        positions = self.positions_for_chi_assignment(assignment)
        floor_nm = DISCRETE_STATE_CONTACT_FLOOR_ANGSTROM / 10.0
        heavy = _heavy_mask(self.topology)
        violations = [(float(np.linalg.norm(positions[i] - positions[j])) * 10.0, i, j)
                      for i, j in self._contact_pairs(positions, floor_nm)]
        violations = [v for v in violations if _impossible_contact(v[0], bool(heavy[v[1]] and heavy[v[2]]))]
        if not violations:
            return None
        atoms = list(self.topology.atoms())
        distance, i, j = min(violations)
        label = lambda k: (f"{atoms[k].residue.chain.id}:{atoms[k].residue.id}:"
                           f"{atoms[k].residue.name}:{atoms[k].name}")
        return dict(distance_angstrom=distance, atoms=[label(i), label(j)])

    def energy_for_chi_assignment(
        self, assignment: Mapping[str, Sequence[float]]
    ) -> float:
        """Amber14 potential for an exact residue->chi1..chiN assignment."""
        return self.energy(self.positions_for_chi_assignment(assignment))

    def build(self) -> QUBOResult:
        """Inclusion-exclusion physical terms; validate full-assignment energy equivalence."""
        # Use a complete candidate assignment as decomposition origin. A heavily
        # clashing perturbed input would otherwise cause catastrophic cancellation.
        count=len(self.candidates)
        admissibility=_discrete_state_admissibility(self)
        env_ok,pair_ok=admissibility["single_ok"],admissibility["pair_ok"]
        rank={v:self.energy(self.positions_for_variables([v])) if env_ok[v] else math.inf
              for v in range(count)}
        try:
            anchors=_admissible_assignment(self.site_to_variables,env_ok,pair_ok,
                                           order=lambda group:sorted(group,key=lambda v:rank[v]))
        except _SearchLimitExceeded:
            anchors=None          # treated as no admissible assignment: raw fallback below
        raw_fallback=anchors is None
        if raw_fallback:
            # Every combination contains an impossible contact, so there is no
            # admissible space to be exact on. Build as before A47 from raw
            # energies; the precision budget below still applies and fails
            # closed if those energies are too large to represent.
            env_ok=np.ones(count,dtype=bool); pair_ok=np.ones((count,count),dtype=bool)
            rank={v:self.energy(self.positions_for_variables([v])) for v in range(count)}
            anchors=[min(group,key=lambda v:rank[v]) for group in self.site_to_variables.values()]
        anchor_positions=self.positions_for_variables(anchors)
        def assignment(variables):
            positions=anchor_positions.copy()
            for v in variables:
                c=self.candidates[v]
                positions[c["indices"]]=c["positions"]
            return positions
        self.decomposition_quality=dict(
            anchor=topology_geometry_audit(self.topology,anchor_positions),
            single_site_candidates=[dict(variable=v,residue_id=c["residue_id"],
                audit=topology_geometry_audit(self.topology,assignment([v])))
                for v,c in enumerate(self.candidates)],
            background="other Active sites fixed at decomposition anchors; not an exhaustive pair-combination audit",
            candidate_filtering=dict(policy="exclude sub-0.4-A topology-excluded intramolecular overlaps before retention",
                                     exclusions=self.candidate_quality_exclusions),
            discrete_state_admissibility=admissibility["audit"])
        baseline=self.energy(anchor_positions)
        # Energies of geometrically impossible states are never evaluated: an
        # r^-12 wall at atom-on-atom contact exceeds 1e11 kcal/mol and would
        # consume the float64 precision of every other coefficient.
        singles=np.zeros(count)
        for v in range(count):
            if env_ok[v]:
                singles[v]=self.energy(assignment([v]))-baseline
        pairs=np.zeros((count,count))
        for i in range(count):
            for j in range(i+1,count):
                if (self.candidates[i]["site"]!=self.candidates[j]["site"]
                        and env_ok[i] and env_ok[j] and pair_ok[i,j]):
                    pairs[i,j]=self.energy(assignment([i,j]))-baseline-singles[i]-singles[j]
        # Every admissible assignment has relative energy within +-bound, so a
        # forbidden state priced at 2*bound+1 is above all of them.
        bound=float(np.abs(singles).sum()+np.abs(pairs).sum())
        forbidden_penalty=2.0*bound+1.0
        for v in range(count):
            if not env_ok[v]:
                singles[v]=forbidden_penalty
        for i in range(count):
            for j in range(i+1,count):
                if (self.candidates[i]["site"]!=self.candidates[j]["site"]
                        and env_ok[i] and env_ok[j] and not pair_ok[i,j]):
                    pairs[i,j]=forbidden_penalty
        helper=InterfaceQUBOBuilder(min_variables=2,max_variables=30)
        penalty=helper._lambda_lower_bound(singles,pairs,self.site_to_variables)
        q=pairs.copy(); np.fill_diagonal(q,singles-penalty)
        for group in self.site_to_variables.values():
            for a,b in _combinations(group): q[a,b]+=2*penalty
        # All-atom candidates come from an mmCIF, not a PyG graph: there is no
        # graph node, so node_index/original_node_index are -1 (residue_id is
        # the identity). prior_probability is the candidate's own Dunbrack
        # (or legacy-override) prior, not a uniform placeholder.
        records=tuple(VariableRecord(v,c["site"],-1,-1,c["residue_id"],
            __import__("gemmi").find_tabulated_residue(c["residue_name"]).one_letter_code,
            list(self.site_to_variables[c["site"]]).index(v),c["angle"],float(c["prior_probability"]),float(singles[v]))
            for v,c in enumerate(self.candidates))
        # Vacuum/NoCutoff Amber14 is exactly pair-decomposable over side-chain
        # choices, so the QUBO must reproduce the full energy (1e-4 kcal/mol)
        # on every admissible assignment; forbidden states carry the penalty.
        # Implicit-solvent GBN2 is not: Born radii depend on every atom, so the
        # same inclusion-exclusion expansion is a pairwise approximation. Its
        # error is measured on the same sampled assignments and recorded, and
        # every structure is still relaxed/scored with the full GBN2 energy.
        exact=self.pair_decomposition_exact
        rng=np.random.default_rng(918); max_error=0.; squared_errors=[]
        for _ in range(12):
            try:
                selected=_admissible_assignment(self.site_to_variables,env_ok,pair_ok,
                                                order=lambda group:list(rng.permutation(group)))
            except _SearchLimitExceeded:
                selected=None
            if selected is None:
                selected=list(anchors)    # an admissible assignment known to exist
            x=np.zeros(count); x[selected]=1
            actual=self.energy(self.positions_for_variables(selected))
            predicted=baseline+singles@x+x@pairs@x
            if not (math.isfinite(actual) and math.isfinite(predicted)):
                raise FloatingPointError("Non-finite all-atom energy during decomposition check")
            max_error=max(max_error,abs(actual-predicted))
            squared_errors.append((actual-predicted)**2)
            if exact and not np.isclose(actual,predicted,atol=1e-4,rtol=1e-9):
                raise ValueError("Force field is not pair-decomposable at required precision")
        rms_error=float(math.sqrt(sum(squared_errors)/len(squared_errors)))
        offset=baseline+penalty*len(self.site_to_variables)
        h,j,ising_offset=qubo_to_ising(q,offset)
        roundoff_bound=ising_roundoff_tolerance(q,offset)
        if roundoff_bound>1e-3:
            raise FloatingPointError(f"All-atom coefficient dynamic range exceeds 0.001 kcal/mol precision budget: {roundoff_bound}")
        ising_error=validate_qubo_ising_equivalence(q,offset,h,j,ising_offset,tolerance=roundoff_bound)
        return QUBOResult(q,records,self.site_to_variables,penalty,penalty,offset,singles,pairs,
            dict(model=("Amber14 all-atom fixed-backbone Dunbrack full chi1..chiN rotamer states"
                   if self.rotamer_mode in ("dunbrack2010","pyrosetta_dun10") and self.chi1_angles_override is None
                   else "Amber14 all-atom fixed-backbone chi1 grid"),energy_unit="kcal/mol",
                physical_constant_offset=baseline,all_atom_equivalence_max_error=max_error,
                all_atom_equivalence_rms_error=rms_error,all_atom_equivalence_samples=12,
                pair_decomposition=("exact" if exact else "pairwise_approximation"),
                candidate_relax_iterations=self.candidate_relax_iterations,
                physical_quality_schema=self.preparation_quality["schema"],
                decomposition_anchor_variables=anchors,ising_equivalence_max_error=ising_error,
                discrete_state_contact_floor_angstrom=DISCRETE_STATE_CONTACT_FLOOR_ANGSTROM,
                forbidden_state_penalty_kcal=forbidden_penalty,
                forbidden_single_states=admissibility["audit"]["forbidden_single_count"],
                forbidden_pair_states=admissibility["audit"]["forbidden_pair_count"],
                equivalence_scope=("raw energies: no geometry-admissible assignment exists" if raw_fallback
                                   else "exact on every geometry-admissible assignment; forbidden states carry the penalty"),
                no_admissible_assignment_raw_fallback=raw_fallback,
                forbidden_variables=[int(v) for v in range(count) if not env_ok[v]],
                forbidden_variable_pairs=[[int(i),int(j)] for i in range(count) for j in range(i+1,count)
                                          if env_ok[i] and env_ok[j] and not pair_ok[i,j]],
                ising_roundoff_tolerance=roundoff_bound,
                atom_count=len(self.base_positions),
                polar_hydrogen_shielding=self.polar_hydrogen_shielding,
                forcefield=(["amber14-all.xml"] if self.solvent_model=="vacuum"
                            else ["amber14-all.xml","implicit/gbn2.xml"]),
                solvent=("vacuum; NoCutoff" if self.solvent_model=="vacuum" else "implicit GBN2; NoCutoff"),
                solvent_model=self.solvent_model,
                raw_rotamer_pool_sizes=self.raw_rotamer_pool_sizes,
                candidate_quality_exclusions=self.candidate_quality_exclusions,
                rotamers_per_site=self.retained_rotamers_per_site,
                site_scores=self.site_scores.tolist(),
                candidate_chi_degrees=[list(candidate.get("chi_degrees",(candidate["angle"],))) for candidate in self.candidates],
                candidate_minimization=[candidate.get("minimization") for candidate in self.candidates],
                rotamer_state_policy=(f"{self.rotamer_mode} full side-chain rotamer states (chi1..chiN) -> 3--6 retained under <=30 variables" if self.rotamer_mode in ("dunbrack2010","pyrosetta_dun10") and self.chi1_angles_override is None else "explicit legacy chi1 angle override"),
                **rotamer_source_metadata(self.rotamer_mode, self.rotamer_library_path),
                candidate_scope=("Dunbrack full side-chain chi state; Amber14 single-candidate prescreen, no affinity claim"
                    if self.rotamer_mode in ("dunbrack2010","pyrosetta_dun10") and self.chi1_angles_override is None
                    else "legacy chi1-only candidate; no affinity claim")))

    def write_structure(self, positions: np.ndarray, destination: Path) -> None:
        """Write author-ID CIF, with occupancy=1 for generated computational atoms."""
        import gemmi
        with Path(destination).open("w") as f:
            self.app.PDBxFile.writeFile(self.topology,positions*self.unit.nanometer,f,keepIds=True)
        structure=gemmi.read_structure(str(destination))
        for chain in structure[0]:
            for residue in chain:
                for atom in residue: atom.occ=1.
        structure.make_mmcif_document().write_file(str(destination))

    def reconstruct(self, bits: Sequence[int], destination: Path, *,
                    minimize_iterations: int = 200) -> dict[str, Any]:
        """Write a full-atom CIF before/after identically constrained local relaxation."""
        x=np.asarray(bits)
        if x.shape!=(len(self.candidates),) or not np.all((x==0)|(x==1)):
            raise ValueError("Invalid binary assignment")
        if any(x[list(g)].sum()!=1 for g in self.site_to_variables.values()):
            raise ValueError("Assignment violates site one-hot constraints")
        positions=self.positions_for_variables(np.flatnonzero(x))
        return self.relax_positions(positions,destination,minimize_iterations=minimize_iterations)

    def perturb_sidechain_chis(
        self, seed: int, min_degrees: float = 40., max_degrees: float = 120.
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Perturb every defined side-chain chi angle without reference-based rejection."""
        if not 0 < min_degrees <= max_degrees <= 180:
            raise ValueError("Require 0 < min_degrees <= max_degrees <= 180")
        rng=np.random.default_rng(seed)
        positions=self.base_positions.copy()
        residue_lookup={}
        bond_graph={i:set() for i in range(self.topology.getNumAtoms())}
        for a,b in self.topology.bonds():
            bond_graph[a.index].add(b.index);bond_graph[b.index].add(a.index)
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            residue_lookup[rid]=residue
        records=[]
        for rid in self.active_residues:
            residue=residue_lookup[rid]
            definitions=_CHI_ATOMS.get(residue.name,())
            if not definitions:
                raise ValueError(f"No supported chi definitions for Active residue {rid}")
            atoms={a.name:a.index for a in residue.atoms()}
            before=[]
            targets=[]
            deltas=[]
            # Read current torsions from the progressively unchanged input,
            # then set all target chis in one sequential internal-coordinate pass.
            for definition in definitions:
                a,b,c,d=(atoms[name] for name in definition)
                current=_torsion_angle_degrees(positions[a],positions[b],positions[c],positions[d])
                delta=float(rng.uniform(min_degrees,max_degrees)*rng.choice([-1,1]))
                before.append(current);deltas.append(delta)
                targets.append(((current+delta+180.0)%360.0)-180.0)
            positions=_apply_sidechain_chis(positions,atoms,bond_graph,residue.name,targets)
            after=[]
            for definition in definitions:
                a,b,c,d=(atoms[name] for name in definition)
                after.append(_torsion_angle_degrees(
                    positions[a],positions[b],positions[c],positions[d]))
            records.append(dict(
                residue_id=rid,seed=seed,residue_name=residue.name,
                chi_before=[float(v) for v in before],
                chi_after=[float(v) for v in after],
                delta_degrees=[float(v) for v in deltas],
            ))
        return positions,records

    def perturb_chi1(self, seed: int, min_degrees: float = 40.,
                     max_degrees: float = 120.) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Deterministic signed chi1 perturbations; no energy/reference-based rejection.

        This is a retrospective fixed-backbone recovery control. Other chi angles
        and the backbone remain input-derived, so it is not de novo prediction.
        """
        if not 0 < min_degrees <= max_degrees <= 180:
            raise ValueError("Require 0 < min_degrees <= max_degrees <= 180")
        rng=np.random.default_rng(seed);positions=self.base_positions.copy();record=[]
        residues={}
        for residue in self.topology.residues():
            residues[f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"]=residue
        for site,rid in enumerate(self.active_residues):
            atoms={a.name:a.index for a in residues[rid].atoms()}
            indices=self.candidates[self.site_to_variables[site][0]]["indices"]
            center=self.base_positions[atoms["CA"]]
            axis=self.base_positions[atoms["CB"]]-center;axis/=np.linalg.norm(axis)
            degrees=float(rng.uniform(min_degrees,max_degrees)*rng.choice([-1,1]))
            angle=np.deg2rad(degrees);relative=self.base_positions[indices]-center
            positions[indices]=center+relative*np.cos(angle)+np.cross(axis,relative)*np.sin(angle)+np.outer(relative@axis,axis)*(1-np.cos(angle))
            old=_chi1_angle({n:self.base_positions[i] for n,i in atoms.items()},residues[rid].name)
            new=_chi1_angle({n:positions[i] for n,i in atoms.items()},residues[rid].name)
            if abs((new-old-degrees+180)%360-180)>1e-6:
                raise AssertionError("Perturbed chi1 does not match requested rotation")
            record.append(dict(residue_id=rid,seed=seed,chi1_before=old,chi1_after=new,delta_degrees=degrees))
        return positions,record

    def relax_cdr_loop(self, destination: Path, cdr_residues: Sequence[str], *,
                       iterations: int = 100, restraint_k: float = 100.) -> dict[str, Any]:
        """Stage 2: loop atoms plus Active sidechains move; weak BB restraint to stage 1.

        k is kJ/mol/nm^2; potential is k/2*distance^2. Loop side chains and H
        move with their backbone to avoid stretching bonds to immobilized atoms.
        """
        if iterations <= 0 or restraint_k <= 0 or not cdr_residues:
            raise ValueError("Stage 2 requires mapped CDR residues and positive controls")
        parsed=self.app.PDBxFile(str(destination))
        positions=np.asarray(parsed.positions.value_in_unit(self.unit.nanometer))
        movable=set(self.movable); backbone=[]; found=set()
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            if rid in cdr_residues:
                found.add(rid)
                for atom in residue.atoms():
                    movable.add(atom.index)
                    if atom.name in ('N','CA','C','O'): backbone.append(atom.index)
        if found!=set(cdr_residues): raise ValueError("CDR residue topology mapping failed")
        system=self.mm.XmlSerializer.deserialize(self.mm.XmlSerializer.serialize(self.system))
        frozen=sorted(set(range(len(positions)))-movable)
        for i in frozen: system.setParticleMass(i,0)
        force=self.mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
        force.addGlobalParameter('k',restraint_k)
        for name in ('x0','y0','z0'): force.addPerParticleParameter(name)
        for i in backbone: force.addParticle(i,positions[i].tolist())
        system.addForce(force)
        integrator=self.mm.VerletIntegrator(.001)
        context=_openmm_context(self.mm, system, integrator)
        context.setPositions(positions*self.unit.nanometer)
        initial=float(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(self.unit.kilocalories_per_mole))
        self.mm.LocalEnergyMinimizer.minimize(context,10.,iterations)
        state=context.getState(getPositions=True,getEnergy=True,getForces=True)
        final=np.asarray(state.getPositions(asNumpy=True).value_in_unit(self.unit.nanometer))
        augmented=float(state.getPotentialEnergy().value_in_unit(self.unit.kilocalories_per_mole))
        force_quality=relaxation_force_audit(
            np.asarray(state.getForces(asNumpy=True).value_in_unit(
                self.unit.kilojoules_per_mole/self.unit.nanometer)),movable,iterations=iterations)
        del context,integrator
        if not np.allclose(final[frozen],positions[frozen],atol=1e-10,rtol=0):
            raise AssertionError('Stage 2 moved frozen atoms')
        if augmented>initial+1e-4: raise ValueError('Stage 2 raised restrained objective')
        self.write_structure(positions,Path(destination).with_name(Path(destination).stem+'_stage1.cif'))
        self.write_structure(final,destination)
        return dict(stage2_iterations=iterations,stage2_restraint_k_kj_mol_nm2=restraint_k,
            stage2_physical_energy_kcal=self.energy(final),stage2_restrained_energy_kcal=augmented,
            stage2_force_quality=force_quality,
            stage2_physical_quality=topology_geometry_audit(self.topology,final),
            stage2_backbone_displacement_angstrom=float(10*np.sqrt(np.mean(np.sum((final[backbone]-positions[backbone])**2,axis=1)))),
            stage2_frozen_atoms=len(frozen))

    def relax_positions(self, positions: np.ndarray, destination: Path, *,
                        minimize_iterations: int = 200) -> dict[str, Any]:
        """Same constrained relaxation for sampled candidates and unsearched input."""
        if minimize_iterations<0: raise ValueError("minimize_iterations must be nonnegative")
        positions=np.asarray(positions,dtype=float).copy()
        if positions.shape!=self.base_positions.shape or not np.isfinite(positions).all():
            raise ValueError("Invalid full-atom positions")
        frozen=sorted(set(range(len(positions)))-self.movable)
        if not np.allclose(positions[frozen],self.base_positions[frozen],atol=1e-10,rtol=0):
            raise ValueError("Input changed frozen/background atoms")
        before=self.energy(positions)
        before_components=self.energy_components()
        before_quality=topology_geometry_audit(self.topology,positions)
        destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
        self.write_structure(positions,destination.with_name(destination.stem+"_discrete.cif"))
        if minimize_iterations:
            system=self.mm.XmlSerializer.deserialize(self.mm.XmlSerializer.serialize(self.system))
            for i in range(len(positions)):
                if i not in self.movable: system.setParticleMass(i,0)
            integrator=self.mm.VerletIntegrator(.001)
            context=_openmm_context(self.mm, system, integrator)
            positions,minimization=_minimize_movable_positions(
                context,positions,self.movable,minimize_iterations,self.unit)
            del context,integrator
        else:
            minimization=dict(minimizer="skipped",minimizer_iterations=0,
                              minimizer_evaluations=0,minimizer_stop_reason="iteration cap zero",
                              minimizer_reported_success=False,
                              minimizer_force_verified_success=False,
                              minimizer_restart_count=0)
        frozen=sorted(set(range(len(positions)))-self.movable)
        if not np.allclose(positions[frozen],self.base_positions[frozen],atol=1e-10,rtol=0):
            raise AssertionError("Frozen atoms moved during relaxation")
        after=self.energy(positions)
        after_components=self.energy_components()
        if after>before+1e-4: raise ValueError("Relaxation increased potential energy")
        self.write_structure(positions,destination)
        audit_forces=_ExactForces(self.context,self.movable)
        _,final_forces=audit_forces(positions,self.unit)
        force_quality=relaxation_force_audit(final_forces,self.movable,iterations=minimize_iterations)
        force_quality["force_audit_platform"]=("Reference" if audit_forces.reference_evaluations
                                               else self.context.getPlatform().getName())
        after_quality=topology_geometry_audit(self.topology,positions)
        return dict(discrete_energy_kcal=before,relaxed_energy_kcal=after,
            max_iterations=minimize_iterations,frozen_atoms=len(frozen),movable_atoms=len(self.movable),
            **minimization,
            **force_quality, physical_quality_before=before_quality,
            physical_quality_after=after_quality,
            discrete_energy_components_kcal=before_components,
            relaxed_energy_components_kcal=after_components,
            note="Iteration cap is not a convergence guarantee; fixed-backbone vacuum energy is not binding affinity")

    def energy_components(self) -> dict[str, float]:
        """Force-group energies at the currently set positions; no energy clipping."""
        return {name:float(self.context.getState(getEnergy=True,groups={group})
            .getPotentialEnergy().value_in_unit(self.unit.kilocalories_per_mole))
            for name,group in self.energy_force_groups.items()}
