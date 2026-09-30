"""Read-only continuation of a saved structural output under its fixed backbone.

This diagnostic does not change a formal run, its acceptance status, or its
saved structures.  It asks whether the existing 200-iteration cap explains a
failed force audit, while also reporting extreme atom overlaps.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from openmm import LocalEnergyMinimizer, Platform, Context, VerletIntegrator, app, unit

from nanoqc.structure.physical_quality import (
    RELAX_FORCE_TOLERANCE_KJ_MOL_NM, relaxation_force_audit,
    topology_geometry_audit,
)


def _atom_keys(topology: app.Topology) -> list[tuple[str, str, str, str]]:
    return [(a.residue.chain.id, a.residue.id, a.residue.insertionCode, a.name)
            for a in topology.atoms()]


def _movable(topology: app.Topology, active_residues: list[str]) -> set[int]:
    active = set(active_residues)
    bonds = {a.index: set() for a in topology.atoms()}
    for left, right in topology.bonds():
        bonds[left.index].add(right.index)
        bonds[right.index].add(left.index)
    movable = set()
    found = set()
    for residue in topology.residues():
        rid = f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
        if rid not in active:
            continue
        found.add(rid)
        atoms = {a.name: a.index for a in residue.atoms()}
        ca, cb = atoms["CA"], atoms["CB"]
        if cb not in bonds[ca]:
            raise ValueError(f"Missing CA-CB bond at {rid}")
        seen = {ca}
        stack = [cb]
        moving = set()
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            moving.add(index)
            stack.extend(bonds[index] - seen)
        if not moving <= set(atoms.values()):
            raise ValueError(f"Side chain traversal escaped residue {rid}")
        movable.update(moving)
    if found != active:
        raise ValueError(f"Active residues missing: {sorted(active-found)}")
    return movable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("relax_only", "qaoa", "sa", "uniform", "greedy"),
                        required=True)
    parser.add_argument("--increments", type=int, nargs="+", default=[200, 800, 1000])
    parser.add_argument("--scipy-iterations", type=int, default=0,
                        help="Also try L-BFGS-B on exactly the movable coordinates")
    args = parser.parse_args()
    if not args.increments or any(n <= 0 for n in args.increments):
        parser.error("--increments must contain positive iteration caps")
    root = args.experiment_dir.resolve()
    case = json.loads((root.parent / "experiment.json").read_text())
    prepared = app.PDBxFile(str(root / "prepared_input.cif"))
    prior = json.loads((root / "run_manifest.json").read_text())
    loop_relax = int(prior.get("arguments", {}).get("loop_relax_iterations", 0))
    stem = "relax_only" if args.method == "relax_only" else f"{args.method}_relaxed"
    saved = app.PDBxFile(str(root / f"{stem}{'_stage1' if loop_relax else ''}.cif"))
    if _atom_keys(prepared.topology) != _atom_keys(saved.topology):
        raise ValueError("Saved and prepared topologies differ")
    reference = np.asarray(prepared.positions.value_in_unit(unit.nanometer), dtype=float)
    positions = np.asarray(saved.positions.value_in_unit(unit.nanometer), dtype=float)
    movable = _movable(prepared.topology, case["active_residues"])
    frozen = sorted(set(range(len(reference))) - movable)
    max_frozen_drift = float(np.max(np.linalg.norm(positions[frozen]-reference[frozen], axis=1)))
    if max_frozen_drift > 0.001:
        raise ValueError(f"Saved frozen atoms differ by {max_frozen_drift*10:.4f} A")
    positions[frozen] = reference[frozen]
    forcefield = app.ForceField("amber14-all.xml")
    system = forcefield.createSystem(prepared.topology, nonbondedMethod=app.NoCutoff,
                                     constraints=None, rigidWater=False, removeCMMotion=False)
    for index in frozen:
        system.setParticleMass(index, 0)
    integrator = VerletIntegrator(.001)
    platform_name = os.environ.get("QP_OPENMM_PLATFORM", "CPU")
    properties = ({"Threads": os.environ.get("OPENMM_CPU_THREADS", "8")}
                  if platform_name == "CPU" else
                  {"Precision": os.environ.get("QP_OPENMM_PRECISION", "double"),
                   "DeviceIndex": os.environ.get("QP_OPENMM_DEVICE", "0")}
                  if platform_name == "CUDA" else {})
    context = Context(system, integrator, Platform.getPlatformByName(platform_name), properties)
    context.setPositions(positions * unit.nanometer)
    if args.scipy_iterations:
        from scipy.optimize import minimize
        moving = np.array(sorted(movable), dtype=int)
        starting = positions.copy()

        def objective(x):
            xyz = starting.copy()
            xyz[moving] = x.reshape(-1, 3)
            context.setPositions(xyz * unit.nanometer)
            state = context.getState(getEnergy=True, getForces=True)
            energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
            force = np.asarray(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer))
            return energy, -force[moving].ravel()

        solution = minimize(objective, positions[moving].ravel(), jac=True,
                            method="L-BFGS-B", options={"maxiter": args.scipy_iterations,
                                                          "ftol": 1e-15, "gtol": 1e-3})
        positions[moving] = solution.x.reshape(-1, 3)
        context.setPositions(positions * unit.nanometer)
        print(json.dumps(dict(scipy_success=bool(solution.success),
                              scipy_message=str(solution.message),
                              scipy_iterations=int(solution.nit),
                              scipy_evaluations=int(solution.nfev))))
    for increment in [0, *args.increments]:
        if increment:
            LocalEnergyMinimizer.minimize(context, RELAX_FORCE_TOLERANCE_KJ_MOL_NM, increment)
        state = context.getState(getPositions=True, getEnergy=True, getForces=True)
        xyz = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer))
        quality = topology_geometry_audit(prepared.topology, xyz)
        force = relaxation_force_audit(forces, movable, iterations=1)
        print(json.dumps(dict(increment=increment,
            energy_kcal=float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)),
            rms_force=force["movable_force_rms_kj_mol_nm"],
            converged=force["relaxation_converged"],
            overlaps=quality["extreme_nonbonded_pair_count"],
            closest_pair=quality["closest_nonbonded_pair"])))
        if force["relaxation_converged"] and quality["geometry_passed"]:
            break


if __name__ == "__main__":
    main()
