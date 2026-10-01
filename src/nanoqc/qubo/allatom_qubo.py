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
from nanoqc.qubo.ising import qubo_to_ising, validate_qubo_ising_equivalence
from nanoqc.qubo.qubo_types import AA_INDEX, QUBOResult, RotamerTemplate, VariableRecord
from nanoqc.qubo.rotamer_library import _THREE_LETTER, _dunbrack_templates_for_site, _expanded_rotamer_templates, _load_rotamer_bins, _nearest_dunbrack_bin, rotamer_source_metadata
from nanoqc.structure.physical_quality import (StructureQualityError, topology_geometry_audit,
    relaxation_force_audit, RELAX_FORCE_TOLERANCE_KJ_MOL_NM,
    EXTREME_NONBONDED_FLOOR_ANGSTROM, DEFAULT_MAX_PERTURBATION_ATTEMPTS,
    generated_input_validity, sample_valid_input)


def _minimize_movable_positions(context: Any, positions: np.ndarray,
                                movable: set[int], max_iterations: int,
                                unit: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize the exact OpenMM potential over movable coordinates only.

    OpenMM's LocalEnergyMinimizer can stop with substantial residual force when
    most particles have zero mass to enforce a fixed backbone.  Explicitly
    optimizing only the declared movable coordinates leaves the frozen atoms
    bitwise fixed and permits an independent force-based acceptance audit.
    """
    from scipy.optimize import minimize

    if max_iterations <= 0 or not movable:
        raise ValueError("Movable minimization needs a positive cap and atom set")
    start = np.asarray(positions, dtype=np.float64)
    if start.ndim != 2 or start.shape[1] != 3 or not np.isfinite(start).all():
        raise ValueError("Movable minimization requires finite [atoms,3] positions")
    indices = np.asarray(sorted(movable), dtype=np.int64)
    if indices[0] < 0 or indices[-1] >= len(start):
        raise ValueError("Movable atom index outside topology")

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        current = start.copy()
        current[indices] = flat.reshape(-1, 3)
        context.setPositions(current * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer), dtype=np.float64)
        gradient = -forces[indices].ravel()
        if not math.isfinite(energy) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite energy or force during movable minimization")
        return energy, gradient

    # Relative objective convergence is unsafe for a severely clashing
    # structure: an enormous potential can change by less than its relative
    # tolerance while the remaining forces are still enormous.  Restarting
    # the quasi-Newton history in bounded stages also gives difficult poses
    # another chance without exceeding the declared iteration budget.
    flat = start[indices].ravel().copy()
    iterations = evaluations = 0
    stops = []
    force_verified_success = False
    solver_reported_success = False
    while iterations < max_iterations:
        allowance = min(200, max_iterations - iterations)
        result = minimize(objective, flat, jac=True, method="L-BFGS-B",
                          options={"maxiter": allowance, "maxls": 50,
                                   "ftol": 0.0, "gtol": 1e-3})
        next_flat = np.asarray(result.x, dtype=np.float64)
        moved = not np.array_equal(next_flat, flat)
        iterations += int(result.nit)
        evaluations += int(result.nfev)
        stops.append(str(result.message))
        solver_reported_success = bool(result.success)
        _, gradient = objective(next_flat)
        evaluations += 1
        force_rms = float(np.sqrt(np.mean(gradient ** 2)))
        flat = next_flat
        if force_rms <= RELAX_FORCE_TOLERANCE_KJ_MOL_NM:
            force_verified_success = True
            break
        # An optimizer can terminate with zero iterations when its line search
        # cannot make progress.  Do not loop indefinitely or call that success.
        if result.nit == 0 or not moved:
            break
    final = start.copy()
    final[indices] = flat.reshape(-1, 3)
    if not np.isfinite(final).all():
        raise FloatingPointError("Nonfinite coordinates after movable minimization")
    context.setPositions(final * unit.nanometer)
    return final, dict(minimizer="scipy_lbfgsb_exact_movable",
                       minimizer_iterations=iterations,
                       minimizer_evaluations=evaluations,
                       minimizer_stop_reason="; ".join(stops),
                       minimizer_reported_success=solver_reported_success,
                       minimizer_force_verified_success=force_verified_success,
                       minimizer_restart_count=max(0, len(stops)-1))


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

    def energy_for_chi_assignment(
        self, assignment: Mapping[str, Sequence[float]]
    ) -> float:
        """Amber14 potential for an exact residue->chi1..chiN assignment."""
        return self.energy(self.positions_for_chi_assignment(assignment))

    def positions_for_chi1_assignment(self, assignment: Mapping[str, float]) -> np.ndarray:
        """Apply explicit chi1 angles to Active residues without candidate-set projection.

        This is used by the TRAIN-ONLY coarse-to-Amber calibration stage so the
        exact same chi1 assignment evaluated by the coarse model is evaluated by
        Amber14, rather than snapping to the all-atom builder's retained states.
        """
        positions=self.base_positions.copy()
        residue_lookup={}
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            residue_lookup[rid]=residue
        for rid,angle in assignment.items():
            if rid not in self.active_residues:
                raise ValueError(f"Calibration assignment contains non-active residue: {rid}")
            residue=residue_lookup[rid]
            atoms={a.name:a.index for a in residue.atoms()}
            ca,cb=atoms["CA"],atoms["CB"]
            site=self.active_residues.index(rid)
            indices=self.candidates[self.site_to_variables[site][0]]["indices"]
            original=_chi1_angle({n:positions[i] for n,i in atoms.items()},residue.name)
            delta=np.deg2rad((float(angle)-original+180)%360-180)
            center=positions[ca]
            axis=positions[cb]-center
            axis/=np.linalg.norm(axis)
            relative=positions[indices]-center
            rotated=(relative*np.cos(delta)+np.cross(axis,relative)*np.sin(delta)
                     +np.outer(relative@axis,axis)*(1-np.cos(delta)))
            positions[indices]=center+rotated
            checked={n:positions[i] for n,i in atoms.items()}
            observed=_chi1_angle(checked,residue.name)
            if abs((observed-float(angle)+180)%360-180)>1e-5:
                raise AssertionError("Explicit chi1 calibration rotation mismatch")
        return positions

    def energy_for_chi1_assignment(self, assignment: Mapping[str, float]) -> float:
        """Amber14 potential for an exact residue->chi1 assignment."""
        return self.energy(self.positions_for_chi1_assignment(assignment))

    def build(self) -> QUBOResult:
        """Inclusion-exclusion physical terms; validate full-assignment energy equivalence."""
        # Use a complete candidate assignment as decomposition origin. A heavily
        # clashing perturbed input would otherwise cause catastrophic cancellation.
        count=len(self.candidates)
        anchors=[min(group,key=lambda v:self.energy(self.positions_for_variables([v])))
                 for group in self.site_to_variables.values()]
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
                                     exclusions=self.candidate_quality_exclusions))
        baseline=self.energy(anchor_positions)
        singles=np.array([self.energy(assignment([v]))-baseline for v in range(count)])
        pairs=np.zeros((count,count))
        for i in range(count):
            for j in range(i+1,count):
                if self.candidates[i]["site"]!=self.candidates[j]["site"]:
                    pairs[i,j]=self.energy(assignment([i,j]))-baseline-singles[i]-singles[j]
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
        # choices, so the QUBO must reproduce the full energy (1e-4 kcal/mol).
        # Implicit-solvent GBN2 is not: Born radii depend on every atom, so the
        # same inclusion-exclusion expansion is a pairwise approximation. Its
        # error is measured on the same sampled assignments and recorded, and
        # every structure is still relaxed/scored with the full GBN2 energy.
        exact=self.pair_decomposition_exact
        rng=np.random.default_rng(918); max_error=0.; squared_errors=[]
        for _ in range(12):
            selected=[int(rng.choice(g)) for g in self.site_to_variables.values()]
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
        roundoff_bound=max(1e-9,32*np.finfo(float).eps*(abs(offset)+np.abs(q).sum()+1))
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
                ising_roundoff_tolerance=roundoff_bound,
                atom_count=len(self.base_positions),
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
        return self._draw_sidechain_chis(np.random.default_rng(seed), seed, min_degrees, max_degrees)

    def _draw_sidechain_chis(self, rng: np.random.Generator, seed: int, min_degrees: float,
                             max_degrees: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """One multi-chi draw from ``rng`` (consumption identical to the former body)."""
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

    def perturb_valid_input(self, mode: str, seed: int, min_degrees: float, max_degrees: float, *,
                            heavy_floor_angstrom: float,
                            max_attempts: int = DEFAULT_MAX_PERTURBATION_ATTEMPTS):
        """A perturbed recovery input that is physically possible (A44).

        Draws come from the unchanged perturbation (``multi_chi`` or ``chi1``)
        and are rejected only when the input violates the protocol's existing
        validity floors; see :func:`physical_quality.sample_valid_input`.
        Returns ``(positions, angle_records, validity_audit)``.
        """
        if not 0 < min_degrees <= max_degrees <= 180:
            raise ValueError("Require 0 < min_degrees <= max_degrees <= 180")
        draws = {"multi_chi": self._draw_sidechain_chis, "chi1": self._draw_chi1}
        if mode not in draws:
            raise ValueError("perturbation mode must be multi_chi or chi1")
        return sample_valid_input(
            lambda rng: draws[mode](rng, seed, min_degrees, max_degrees),
            lambda positions: generated_input_validity(
                self.topology, positions, heavy_floor_angstrom=heavy_floor_angstrom),
            seed, max_attempts=max_attempts)

    def perturb_chi1(self, seed: int, min_degrees: float = 40.,
                     max_degrees: float = 120.) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Deterministic signed chi1 perturbations; no energy/reference-based rejection.

        This is a retrospective fixed-backbone recovery control. Other chi angles
        and the backbone remain input-derived, so it is not de novo prediction.
        """
        if not 0 < min_degrees <= max_degrees <= 180:
            raise ValueError("Require 0 < min_degrees <= max_degrees <= 180")
        return self._draw_chi1(np.random.default_rng(seed), seed, min_degrees, max_degrees)

    def _draw_chi1(self, rng: np.random.Generator, seed: int, min_degrees: float,
                   max_degrees: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """One chi1 draw from ``rng`` (consumption identical to the former body)."""
        positions=self.base_positions.copy();record=[]
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
        force_state=self.context.getState(getForces=True)
        force_quality=relaxation_force_audit(
            np.asarray(force_state.getForces(asNumpy=True).value_in_unit(
                self.unit.kilojoules_per_mole/self.unit.nanometer)),
            self.movable, iterations=minimize_iterations)
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
