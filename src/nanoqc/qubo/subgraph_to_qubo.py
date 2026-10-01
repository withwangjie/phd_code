"""Map a pruned protein-interface graph to an upper-triangular QUBO matrix.

The delivered PyG graphs contain one CA coordinate per residue, not complete
backbone/side-chain atoms. This module therefore implements a deterministic,
coarse-grained rotamer model rather than claiming atomistic Dunbrack accuracy:

* each selected VHH residue first receives a 6--12-state chi1 sub-rotamer pool;
* prior, frozen-environment and antigen-guidance energies pre-screen that pool;
* 3--6 states per residue enter the QUBO under a global <=30-variable budget;
* a local frame derived from neighbouring CA and ligand directions attaches
  residue-specific side-chain pseudo-atoms;
* softened, truncated Lennard-Jones and distance-dependent dielectric Coulomb
  terms score rotamer-environment and rotamer-rotamer interactions.

Energies are in approximate kcal/mol units. The model is suitable for QUBO
algorithm development and NISQ-scale experiments, but should be calibrated or
replaced by an all-atom force field before quantitative affinity claims.

QUBO convention
---------------
``Q`` is upper triangular, including its diagonal, and the binary objective is

``E(x) = constant_offset + sum_i Q[i,i] x_i + sum_{i<j} Q[i,j] x_i x_j``.

This is also equal to ``constant_offset + x.T @ Q @ x`` because the lower
triangle is exactly zero. One-hot penalties contribute ``-lambda`` to each
site-variable diagonal, ``+2*lambda`` between rotamers of the same site, and
``+lambda`` per site to ``constant_offset``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nanoqc.data.safe_graph_load import load_graph

# Names from the split-out modules that this module or its callers use.
from nanoqc.qubo.atomistic_structure import (  # noqa: E402,F401
    _SIDECHAIN_NAMES,
    _SYMMETRIC_SWAPS,
    read_atomistic_structure,
    _backbone_phi_psi,
    _sidechain_chi_angles,
    _apply_sidechain_chis,
    evaluate_atomistic_prediction,
)
from nanoqc.qubo.qubo_types import (  # noqa: E402,F401
    AA_INDEX,
    ForceFieldConfig,
    EnergyCalibration,
    RotamerTemplate,
    RotamerState,
    chi1_well_index,
    select_chi1_well_representatives,
)
from nanoqc.qubo.rotamer_library import (  # noqa: E402,F401
    _THREE_LETTER,
    _nearest_dunbrack_bin,
    _load_dunbrack_bins,
    PyRosettaRotamerProvider,
    _load_rotamer_bins,
    rotamer_source_metadata,
    _dunbrack_templates_for_site,
)
from nanoqc.qubo.coarse_qubo import (  # noqa: E402,F401
    InterfaceQUBOBuilder,
)
from nanoqc.qubo.ising import (  # noqa: E402,F401
    qubo_to_ising,
    validate_qubo_ising_equivalence,
)
from nanoqc.qubo.allatom_qubo import (  # noqa: E402,F401
    _virtual_pruned_graph,
    AllAtomInterfaceQUBOBuilder,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        help="Optional trusted local pruned PyG .pt file; defaults to a virtual graph.",
    )
    arguments = parser.parse_args()
    if arguments.graph is None:
        example = _virtual_pruned_graph(site_count=6)
    else:
        # torch.load uses pickle for PyG Data. Only load files generated locally
        # or obtained from a trusted source.
        example = load_graph(arguments.graph)

    builder = InterfaceQUBOBuilder()
    result = builder.build(example)
    Q = result.Q
    assert Q.shape[0] == Q.shape[1] <= 30
    assert Q.shape[0] >= 20
    assert np.count_nonzero(np.tril(Q, k=-1)) == 0
    assert np.isfinite(Q).all()
    assert all(3 <= len(indices) <= 6 for indices in result.site_to_variables.values())
    assert result.lambda_value >= result.lambda_lower_bound
    assert result.metadata["lambda_exceeds_max_pair_attraction"]

    h, J, offset = qubo_to_ising(Q, result.constant_offset)
    assert validate_qubo_ising_equivalence(
        Q, result.constant_offset, h, J, offset
    ) <= 1e-9
    one_hot = np.zeros(Q.shape[0], dtype=np.int8)
    for variables in result.site_to_variables.values():
        one_hot[variables[0]] = 1
    spins = 1.0 - 2.0 * one_hot
    qubo_energy = result.energy(one_hot)
    ising_energy = float(offset + h @ spins)
    for left in range(len(spins)):
        for right in range(left + 1, len(spins)):
            ising_energy += J[left, right] * spins[left] * spins[right]
    assert np.isclose(qubo_energy, ising_energy, atol=1e-8)

    print("Variable mapping")
    print("idx  site  node  residue  aa  rot  chi1  prior    E_self")
    for record in result.variable_map:
        print(
            f"{record.variable_index:>3}  {record.site_index:>4}  "
            f"{record.node_index:>4}  {record.residue_id:<7}  "
            f"{record.amino_acid:>2}  {record.rotamer_index:>3}  "
            f"{record.chi1_degrees:>5.0f}  {record.prior_probability:>5.2f}  "
            f"{record.self_energy:>9.3f}"
        )
    linear_terms = int(np.count_nonzero(np.abs(h) > 1e-12))
    coupling_terms = int(np.count_nonzero(np.abs(np.triu(J, 1)) > 1e-12))
    print(
        json.dumps(
            {
                "qubo_dimension": int(Q.shape[0]),
                "sites": len(result.site_to_variables),
                "lambda_lower_bound": result.lambda_lower_bound,
                "lambda_used": result.lambda_value,
                "ising_linear_terms": linear_terms,
                "ising_coupling_terms": coupling_terms,
                "basis_hamiltonian_terms": linear_terms + coupling_terms,
                "offset": offset,
                "qubo_ising_energy_check": qubo_energy,
            },
            indent=2,
        )
    )
