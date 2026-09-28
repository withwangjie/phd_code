"""Full-atom nanobody-antigen complex evaluator for the side-chain-optimization /
complex-prediction pipeline (Gemmi-based).

This module supersedes the earlier ``evaluate_complex_dockq`` evaluator (no
longer in this repository) with the three corrections
raised in the academic pre-review (R1-M5): (1) an explicit, non-official
receptor-aligned DockQ field name so it is never confused with the published
DockQ definition; (2) a symmetry-corrected side-chain heavy-atom RMSD for the
caller-declared Active (redesigned) residues, so a physically meaningless
180-degree ring/carboxylate flip does not register as a large false error;
(3) side-by-side multi-stage trajectory evaluation, so a pipeline run's
perturbed input, relax-only control, discretely-picked rotamer structure, and
both post-relaxation stages can be reported together instead of only the
final structure.

Reference frame and alignment
------------------------------
The rigid-body superposition (Kabsch algorithm) that brings a predicted
structure into the reference frame is fit **only** on the antigen
(receptor)'s backbone heavy atoms (N, CA, C, O), over receptor residues that
are *not* in the caller-declared ``active_residues`` set (in this project's
usual convention ``active_residues`` are the handful of nanobody paratope
residues being redesigned, so this exclusion is a no-op on the receptor side
and the fit uses the whole receptor backbone; the exclusion is still applied
generally in case a future active set ever includes receptor residues, e.g.
a locally flexible epitope patch). The nanobody (ligand) backbone is never
used to compute this fit -- fitting it would let a global superposition
partially explain away a real docking-pose error. It is still *transformed*
by the receptor-derived rotation/translation like every other predicted
atom, which is the entire point: LRMSD and interface metrics must reflect
the predicted pose in the receptor's own frame, not the ligand's self-best
overlay.

Metric definitions
-------------------
* **Fnat**: fraction of reference receptor-ligand residue-pair contacts
  (heavy-atom pair distance < ``fnat_cutoff``, default 5.0 A) recovered in
  the predicted structure. Computed independently in each structure's own
  coordinate frame -- internal (including inter-chain) distances are
  unaffected by a rigid-body transform of the whole structure, so Fnat needs
  no prior alignment.
* **iRMSD**: all-heavy-atom RMSD, after receptor-frame alignment, of every
  reference residue (either chain) within ``interface_cutoff`` (default
  10.0 A) of the other chain.
* **LRMSD**: nanobody backbone (N, CA, C, O) RMSD after receptor-frame
  alignment.
* **dockq_receptor_aligned_variant**:
  ``(Fnat + 1/(1+(iRMSD/1.5)**2) + 1/(1+(LRMSD/8.5)**2)) / 3``, categorized
  Incorrect (< 0.23), Acceptable ([0.23, 0.49)), Medium ([0.49, 0.80)),
  High (>= 0.80). The field is deliberately **not** named ``dockq_score``:
  this is a single receptor-only rigid-body fit with all-heavy-atom interface
  RMSD, not the original DockQ paper's separate least-squares fit of only
  the interface backbone atoms of both partners together, so it must never
  be silently compared to published-DockQ numbers. ``dockq_definition``
  in the result carries this caveat as text alongside the score.
* **active_sidechain_rmsd_angstrom**: symmetry-corrected side-chain
  heavy-atom RMSD over the caller-declared ``active_residues``, computed
  after the same receptor-frame alignment. For PHE and TYR, the aromatic
  ring's two-fold symmetry means a 180-degree flip about the CB-CG-CZ axis
  swaps CD1<->CD2 *and* CE1<->CE2 *simultaneously* (they are not
  independent degrees of freedom); this module therefore compares the
  unswapped assignment against exactly that one physically valid
  simultaneous double-swap and keeps whichever gives the lower squared
  error, per residue, exactly as :data:`_SYMMETRIC_SWAPS` in
  ``subgraph_to_qubo.py`` already does for chi1 recovery scoring elsewhere
  in this pipeline. The same table's other entries (ASP/GLU carboxylate
  O-O, ARG guanidinium N-N) are genuine symmetries and are treated the same
  way. VAL CG1/CG2 and LEU CD1/CD2 are prochiral (stereochemically distinct)
  and are never swapped.

Stereochemical severe-clash filter
------------------------------------
Severe clashes are counted over every non-covalently-bonded heavy-atom pair
(``clash_cutoff`` default 1.5 A) among the scored receptor+ligand residues.
Bond adjacency -- both consecutive peptide C(i)-N(i+1) pairs and CYS-CYS
disulfide SG-SG pairs -- is determined once from the **reference**
structure's own geometry (not the possibly-distorted predicted/intermediate
structure being scanned), matching this project's ``structural_quality.py``
convention; the same excluded atom-name pairs are then applied whichever
structure is scanned. This is a plain geometric distance filter, not
MolProbity clashscore: there are no van der Waals radii and no general
bonded/1-4 exclusion table beyond the two exclusions named above.

Multi-stage trajectory evaluation
------------------------------------
:func:`evaluate_trajectory` accepts an ordered mapping of stage name to
predicted-structure path (e.g. ``{"perturbed_input": ..., "relax_only":
..., "discrete_picked": ..., "stage1_relaxed": ..., "stage2_relaxed":
...}`` -- this project's canonical five pipeline checkpoints, though any
stage names and count are accepted) and evaluates every stage against the
same single reference with :func:`evaluate_complex_metrics`, returning one
full result dict per stage (tagged with ``"stage"``/``"stage_index"``) plus
stage-over-stage deltas for the headline metrics, so a full R1-M5-style
side-by-side trajectory table can be built without re-parsing the reference
once per stage.

Defensive conventions
----------------------
* Every coordinate array this module hands back or stores internally is a
  read-only numpy view (atom-level write protection): parsing and alignment
  never mutate a previously extracted array in place, they only ever build
  fresh ones.
* Residue-identity and heavy-atom-set mismatches between the reference and a
  prediction raise ``ValueError`` immediately rather than silently
  subsetting or dropping atoms, which could otherwise bias an RMSD without
  any visible symptom.
* Every returned result dict contains only JSON-serializable native Python
  types (``float``/``int``/``str``/``bool``/``list``/``dict``/``None``),
  ready for ``json.dump`` or one flattened CSV row per structure/stage (see
  :func:`to_json` / :func:`append_to_csv` / :func:`append_trajectory_to_csv`).
* This module is intentionally free of exotic dependencies: pure Python +
  numpy + scipy (``cKDTree``) + gemmi. It does not import ``subgraph_to_qubo``
  (which pulls in ``torch``/``torch_geometric`` at import time); the residue
  atom tables both modules need live in the dependency-free
  ``residue_tables`` module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


from nanoqc.structure.residue_tables import BACKBONE_ATOMS

# Definitions now live in focused modules; re-exported so every existing
# `from nanoqc.structure.evaluate_complex_metrics import ...` keeps working.
from nanoqc.structure.complex_atoms import (  # noqa: E402,F401
    PathLike,
    ResidueAtoms,
    StructureAtoms,
    AtomCompletenessError,
    _clean_code,
    _residue_id,
    _frozen_vector,
    read_structure_atoms,
    kabsch_fit,
    _paired_coordinates,
    _rmsd,
    _residue_atom_triples,
    compute_contact_residue_pairs,
    compute_fnat,
    compute_interface_residues,
    _chain_residue_ids,
    verify_full_chain_heavy_atom_completeness,
)
from nanoqc.structure.side_chain_metrics import (  # noqa: E402,F401
    _SIDECHAIN_HEAVY_ATOMS,
    _SYMMETRIC_SWAPS,
    _active_side_chain_error,
    compute_active_side_chain_rmsd,
    ClashPair,
    compute_bonded_exclusions,
    find_severe_clashes,
    compute_dockq,
)


# --------------------------------------------------------------------------
# Top-level single-structure orchestrator
# --------------------------------------------------------------------------

def evaluate_complex_metrics(
    ref_path: PathLike,
    pred_path: PathLike,
    *,
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    active_residues: Sequence[str] = (),
    fnat_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    clash_cutoff: float = 1.5,
    peptide_bond_cutoff: float = 1.9,
    disulfide_cutoff: float = 2.3,
    model_index: int = 0,
) -> Dict[str, Any]:
    """Evaluate one predicted nanobody-antigen complex against a reference.

    See the module docstring for the alignment convention, metric
    definitions, and clash-exclusion convention. Nothing this function reads
    is ever mutated in place: parsing, alignment, and every downstream
    computation build fresh arrays (see :func:`read_structure_atoms`'s
    atom-level write protection).

    A thin file-reading wrapper around :func:`evaluate_complex_metrics_from_atoms`
    -- reads both structures with :func:`read_structure_atoms`, then delegates.
    Callers that already have parsed ``StructureAtoms`` in memory (e.g. a
    caller that also needs those atoms for its own, separate computation)
    should call :func:`evaluate_complex_metrics_from_atoms` directly instead,
    to avoid parsing the same files twice.

    Args:
        ref_path: Reference structure (mmCIF or PDB; format sniffed by
            gemmi from content, not the file extension).
        pred_path: Predicted / relaxed structure, same format flexibility.
        receptor_chains: Antigen chain IDs (as they appear in the file).
        ligand_chains: Nanobody chain IDs. Disjoint from
            ``receptor_chains``; never used to compute the alignment fit.
        active_residues: Residue ids (``"CHAIN:SEQID[ICODE]"``, the same
            convention :func:`read_structure_atoms` produces; this
            project's usual convention is a handful of nanobody paratope
            residues) to score with the symmetry-corrected side-chain RMSD,
            and to EXCLUDE from the alignment fit wherever they happen to
            fall on the receptor. Must be a subset of the receptor+ligand
            residues. Defaults to none.
        fnat_cutoff: Contact distance for Fnat (default 5.0 A).
        interface_cutoff: Distance defining interface residues for iRMSD
            (default 10.0 A).
        clash_cutoff: Severe-clash distance threshold (default 1.5 A).
        peptide_bond_cutoff: C(i)-N(i+1) distance below which a consecutive
            residue pair is treated as a genuine peptide bond and excluded
            from clash scanning (default 1.9 A).
        disulfide_cutoff: SG-SG distance below which a CYS pair is treated
            as a genuine disulfide bond and excluded from clash scanning
            (default 2.3 A).
        model_index: Which model to read from each file (default 0).

    Returns:
        A JSON-serializable dict with every metric described in the module
        docstring, plus alignment/clash provenance and the exact cutoffs
        and file paths used, so a saved result is fully self-describing.
        See the source for the complete key list.

    Raises:
        ValueError: On any invalid argument, chain/residue selection that
            does not exist or is not disjoint as required, a residue-
            identity or heavy-atom-set mismatch between the reference and
            prediction, or too few non-Active receptor residues to fit a
            rotation (fewer than 3).
    """
    ref_atoms = read_structure_atoms(ref_path, model_index=model_index)
    pred_atoms = read_structure_atoms(pred_path, model_index=model_index)
    return evaluate_complex_metrics_from_atoms(
        ref_atoms, pred_atoms,
        receptor_chains=receptor_chains, ligand_chains=ligand_chains,
        active_residues=active_residues, fnat_cutoff=fnat_cutoff,
        interface_cutoff=interface_cutoff, clash_cutoff=clash_cutoff,
        peptide_bond_cutoff=peptide_bond_cutoff, disulfide_cutoff=disulfide_cutoff,
        model_index=model_index, reference_path=ref_path, predicted_path=pred_path,
    )


def evaluate_complex_metrics_from_atoms(
    ref_atoms: StructureAtoms,
    pred_atoms: StructureAtoms,
    *,
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    active_residues: Sequence[str] = (),
    fnat_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    clash_cutoff: float = 1.5,
    peptide_bond_cutoff: float = 1.9,
    disulfide_cutoff: float = 2.3,
    model_index: int = 0,
    reference_path: Optional[PathLike] = None,
    predicted_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Evaluate one predicted complex against a reference, from already-parsed atoms.

    The actual computation body of :func:`evaluate_complex_metrics` (which is
    a thin ``read_structure_atoms`` + delegate wrapper around this function).
    Call this directly when the caller already has both structures parsed as
    ``StructureAtoms`` (e.g. produced by its own, separately-configured
    parser, or by a synthetic test fixture) -- this avoids a second,
    redundant file parse, and does not require ``ref_path``/``pred_path`` to
    be real, readable files.

    Args:
        ref_atoms, pred_atoms: Already-parsed reference/prediction
            structures, in the same shape :func:`read_structure_atoms`
            returns (residue id -> ``{"chain", "seqid", "icode", "name",
            "altloc", "atoms"}``).
        receptor_chains, ligand_chains, active_residues, fnat_cutoff,
            interface_cutoff, clash_cutoff, peptide_bond_cutoff,
            disulfide_cutoff: Same as :func:`evaluate_complex_metrics`.
        model_index: Recorded in the result dict for provenance only (no
            parsing happens here); pass through the value actually used to
            produce ``ref_atoms``/``pred_atoms``, if any (default 0).
        reference_path, predicted_path: Recorded in the result dict's
            ``"reference_path"``/``"predicted_path"`` fields for provenance
            only, if available (``None`` when there is no real file, e.g. a
            synthetic in-memory fixture).

    Returns, Raises: Same as :func:`evaluate_complex_metrics`.
    """
    cutoffs = (fnat_cutoff, interface_cutoff, clash_cutoff, peptide_bond_cutoff, disulfide_cutoff)
    if any((not math.isfinite(v)) or v <= 0 for v in cutoffs):
        raise ValueError(
            "fnat_cutoff, interface_cutoff, clash_cutoff, peptide_bond_cutoff and "
            "disulfide_cutoff must all be positive and finite"
        )

    receptor_chains = list(dict.fromkeys(receptor_chains))
    ligand_chains = list(dict.fromkeys(ligand_chains))
    if not receptor_chains or not ligand_chains:
        raise ValueError("receptor_chains and ligand_chains must both be nonempty")
    if set(receptor_chains) & set(ligand_chains):
        raise ValueError("receptor_chains and ligand_chains must be disjoint")

    receptor_ids = _chain_residue_ids(ref_atoms, receptor_chains)
    ligand_ids = _chain_residue_ids(ref_atoms, ligand_chains)
    if not receptor_ids:
        raise ValueError(f"No residues found in the reference for receptor_chains={receptor_chains}")
    if not ligand_ids:
        raise ValueError(f"No residues found in the reference for ligand_chains={ligand_chains}")

    active_ids = list(dict.fromkeys(active_residues))
    complex_ids = set(receptor_ids) | set(ligand_ids)
    unknown_active = [rid for rid in active_ids if rid not in complex_ids]
    if unknown_active:
        raise ValueError(
            f"active_residues must be a subset of receptor_chains/ligand_chains residues; "
            f"unknown ids: {unknown_active}"
        )

    # (P2 remediation) Full-chain, residue-by-residue heavy-atom completeness,
    # across EVERY evaluated chain -- not just the backbone/interface/Active
    # atoms each downstream computation happens to read. Must run before
    # alignment or any metric trusts either structure's atom content; see
    # verify_full_chain_heavy_atom_completeness's docstring for the exact
    # bypass this closes.
    verify_full_chain_heavy_atom_completeness(
        ref_atoms, pred_atoms, list(receptor_chains) + list(ligand_chains),
    )

    # --- (1) receptor-frame Kabsch alignment, fit only on non-Active receptor backbone ---
    active_on_receptor = set(active_ids) & set(receptor_ids)
    alignment_ids = [rid for rid in receptor_ids if rid not in active_on_receptor]
    if len(alignment_ids) < 3:
        raise ValueError(
            f"Only {len(alignment_ids)} non-Active receptor residue(s) available for alignment; "
            "need at least 3. Widen receptor_chains or shrink active_residues."
        )
    moving_alignment, fixed_alignment = _paired_coordinates(ref_atoms, pred_atoms, alignment_ids, BACKBONE_ATOMS)
    rotation, translation = kabsch_fit(moving_alignment, fixed_alignment)
    alignment_rmsd = _rmsd(moving_alignment @ rotation + translation, fixed_alignment)

    # Apply the SAME rigid transform to every scored predicted atom -- this
    # is a transform of the ligand's coordinates, never a fit of them (see
    # the module docstring). It never touches the reference, and never
    # touches pred_atoms in place -- it builds an independent copy so
    # pred_atoms stays exactly as parsed.
    aligned_pred: StructureAtoms = {}
    for rid in receptor_ids + ligand_ids:
        entry = pred_atoms[rid]
        transformed = {
            name: _frozen_vector(*(xyz @ rotation + translation))
            for name, xyz in entry["atoms"].items()
        }
        aligned_pred[rid] = {**entry, "atoms": transformed}

    # --- (2a) Fnat: frame-independent, computed on the raw (unaligned) prediction ---
    fnat, native_contacts, predicted_contacts = compute_fnat(
        ref_atoms, pred_atoms, receptor_ids, ligand_ids, fnat_cutoff,
    )

    # --- (2b) iRMSD: all heavy atoms of reference interface residues ---
    interface_ids = compute_interface_residues(ref_atoms, receptor_ids, ligand_ids, interface_cutoff)
    irmsd: Optional[float] = None
    if interface_ids:
        moving_interface, fixed_interface = _paired_coordinates(ref_atoms, aligned_pred, interface_ids)
        irmsd = _rmsd(moving_interface, fixed_interface)

    # --- (2c) LRMSD: ligand backbone only ---
    moving_ligand, fixed_ligand = _paired_coordinates(ref_atoms, aligned_pred, ligand_ids, BACKBONE_ATOMS)
    lrmsd = _rmsd(moving_ligand, fixed_ligand)

    # --- (2d) receptor-aligned DockQ-style combination ---
    dockq_score, dockq_category = compute_dockq(fnat, irmsd, lrmsd)

    # --- (3) symmetry-corrected Active side-chain RMSD ---
    active_sidechain_rmsd, active_details = compute_active_side_chain_rmsd(
        ref_atoms, aligned_pred, active_ids,
    )

    # --- (4) severe-clash filter on the predicted structure ---
    # Bond adjacency is fixed from the REFERENCE's own geometry (see
    # compute_bonded_exclusions), then applied while scanning the
    # prediction. A rigid transform of the whole structure preserves every
    # internal pairwise distance, so scanning the raw (unaligned) prediction
    # gives identical clash geometry to scanning aligned_pred, at lower cost.
    scored_ids = receptor_ids + ligand_ids
    exclusions = compute_bonded_exclusions(
        ref_atoms, scored_ids,
        peptide_bond_cutoff=peptide_bond_cutoff, disulfide_cutoff=disulfide_cutoff,
    )
    clashes = find_severe_clashes(pred_atoms, scored_ids, exclusions, clash_cutoff=clash_cutoff)

    # --- (5) standard, JSON-serializable result dict ---
    return {
        "dockq_receptor_aligned_variant": dockq_score,
        "dockq_category": dockq_category,
        "dockq_definition": (
            "dockq_receptor_aligned_variant = (Fnat + 1/(1+(iRMSD/1.5)^2) + "
            "1/(1+(LRMSD/8.5)^2)) / 3; based on a single receptor-only rigid-body "
            "fit and all-heavy-atom interface RMSD, NOT the original DockQ "
            "paper's separate interface-backbone superposition of both "
            "partners -- not directly comparable to published DockQ numbers."
        ),
        "fnat": fnat,
        "irmsd_angstrom": irmsd,
        "lrmsd_angstrom": lrmsd,
        "active_sidechain_rmsd_angstrom": active_sidechain_rmsd,
        "active_sidechain_definition": (
            "Symmetry-corrected side-chain heavy-atom RMSD over active_residues "
            "after receptor-frame alignment; PHE/TYR ring CD1/CD2+CE1/CE2 and "
            "other _SYMMETRIC_SWAPS entries are exhaustively compared as one "
            "simultaneous swap vs. the unswapped assignment, minimum kept per residue."
        ),
        "active_sidechain_per_residue": active_details,
        "native_contacts": len(native_contacts),
        "predicted_contacts": len(predicted_contacts),
        "recovered_contacts": len(native_contacts & predicted_contacts),
        "interface_residue_count": len(interface_ids),
        "interface_residues": list(interface_ids),
        "alignment_residue_count": len(alignment_ids),
        "alignment_atom_count": int(moving_alignment.shape[0]),
        "alignment_rmsd_angstrom": alignment_rmsd,
        "alignment_rotation": rotation.tolist(),
        "alignment_translation": translation.tolist(),
        "num_severe_clashes": len(clashes),
        "has_severe_clash": bool(clashes),
        "severe_clash_pairs": [
            {
                "residue_a": c.residue_a, "atom_a": c.atom_a,
                "residue_b": c.residue_b, "atom_b": c.atom_b,
                "distance_angstrom": c.distance_angstrom,
            }
            for c in clashes
        ],
        "clash_definition": (
            f"heavy-atom distance < {clash_cutoff} A; same-residue pairs, "
            "consecutive peptide-bonded C(i)-N(i+1) pairs, and CYS-CYS "
            "disulfide SG-SG pairs excluded (bond adjacency determined from "
            f"the reference structure's own geometry, C-N < {peptide_bond_cutoff} A, "
            f"SG-SG < {disulfide_cutoff} A); not MolProbity clashscore"
        ),
        "receptor_chains": list(receptor_chains),
        "ligand_chains": list(ligand_chains),
        "active_residues": sorted(
            active_ids,
            key=lambda r: (ref_atoms[r]["chain"], ref_atoms[r]["seqid"], ref_atoms[r]["icode"]),
        ),
        "receptor_residue_count": len(receptor_ids),
        "ligand_residue_count": len(ligand_ids),
        "fnat_cutoff_angstrom": fnat_cutoff,
        "interface_cutoff_angstrom": interface_cutoff,
        "clash_cutoff_angstrom": clash_cutoff,
        "peptide_bond_cutoff_angstrom": peptide_bond_cutoff,
        "disulfide_cutoff_angstrom": disulfide_cutoff,
        "model_index": model_index,
        "reference_path": str(reference_path) if reference_path is not None else None,
        "predicted_path": str(predicted_path) if predicted_path is not None else None,
    }


# --------------------------------------------------------------------------
# Multi-stage trajectory evaluation
# --------------------------------------------------------------------------

# Headline metrics tracked stage-over-stage in evaluate_trajectory's deltas.
_TRAJECTORY_HEADLINE_METRICS: Tuple[str, ...] = (
    "active_sidechain_rmsd_angstrom", "irmsd_angstrom", "fnat", "lrmsd_angstrom",
    "num_severe_clashes", "dockq_receptor_aligned_variant",
)


def evaluate_trajectory(
    ref_path: PathLike,
    stage_paths: Mapping[str, PathLike],
    *,
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    active_residues: Sequence[str] = (),
    fnat_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    clash_cutoff: float = 1.5,
    peptide_bond_cutoff: float = 1.9,
    disulfide_cutoff: float = 2.3,
    model_index: int = 0,
) -> Dict[str, Any]:
    """Evaluate every pipeline stage of one complex against the same reference.

    This project's canonical five checkpoints are ``perturbed_input``,
    ``relax_only``, ``discrete_picked``, ``stage1_relaxed`` and
    ``stage2_relaxed`` (see the module docstring), but ``stage_paths`` may
    contain any number of stages under any names; results are reported in
    the order ``stage_paths`` iterates (a plain ``dict`` preserves insertion
    order in Python).

    Args:
        ref_path: Reference structure, shared across every stage.
        stage_paths: Ordered mapping of stage name to that stage's predicted
            structure path.
        receptor_chains, ligand_chains, active_residues, fnat_cutoff,
            interface_cutoff, clash_cutoff, peptide_bond_cutoff,
            disulfide_cutoff, model_index: Forwarded unchanged to
            :func:`evaluate_complex_metrics` for every stage.

    Returns:
        A JSON-serializable dict with:

        * ``"stages"``: a list of per-stage result dicts, each the full
          :func:`evaluate_complex_metrics` output plus ``"stage"`` and
          ``"stage_index"`` (0-based, in iteration order).
        * ``"stage_order"``: the stage names in evaluated order.
        * ``"reference_path"``.

        Every stage dict also carries ``"delta_vs_previous_stage"``: a dict
        of ``{metric_name: current - previous}`` for every metric in
        :data:`_TRAJECTORY_HEADLINE_METRICS` where both this and the
        previous stage have a non-``None`` value (``num_severe_clashes``
        deltas are ``int``, the rest ``float``); a metric missing from
        either stage is simply omitted rather than raising. The first
        stage's ``"delta_vs_previous_stage"`` is always ``{}`` (no previous
        stage to compare against) -- present rather than omitted, so every
        stage dict shares one schema (this matters for
        :func:`append_trajectory_to_csv`, where all stages of one
        trajectory append to the same CSV columns).

    Raises:
        ValueError: If ``stage_paths`` is empty, or (propagated from
            :func:`evaluate_complex_metrics`) on any per-stage evaluation
            failure -- including which stage failed in the exception message.
    """
    if not stage_paths:
        raise ValueError("stage_paths must contain at least one stage")
    stages: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None
    for index, (stage_name, path) in enumerate(stage_paths.items()):
        try:
            result = evaluate_complex_metrics(
                ref_path, path,
                receptor_chains=receptor_chains, ligand_chains=ligand_chains,
                active_residues=active_residues, fnat_cutoff=fnat_cutoff,
                interface_cutoff=interface_cutoff, clash_cutoff=clash_cutoff,
                peptide_bond_cutoff=peptide_bond_cutoff, disulfide_cutoff=disulfide_cutoff,
                model_index=model_index,
            )
        except ValueError as exc:
            raise ValueError(f"Trajectory stage {stage_name!r} failed: {exc}") from exc
        result["stage"] = stage_name
        result["stage_index"] = index
        # Always present (empty for the first stage), so every stage dict
        # shares one schema -- e.g. for append_trajectory_to_csv, where all
        # stages of one trajectory must append to the same CSV columns.
        deltas: Dict[str, Any] = {}
        if previous is not None:
            for metric in _TRAJECTORY_HEADLINE_METRICS:
                current_value, previous_value = result.get(metric), previous.get(metric)
                if current_value is None or previous_value is None:
                    continue
                deltas[metric] = current_value - previous_value
        result["delta_vs_previous_stage"] = deltas
        stages.append(result)
        previous = result
    return {
        "stages": stages,
        "stage_order": list(stage_paths.keys()),
        "reference_path": str(ref_path),
    }


# --------------------------------------------------------------------------
# Export: standalone JSON or an appended CSV row
# --------------------------------------------------------------------------

def to_json(result: Mapping[str, Any], path: PathLike, *, indent: int = 2) -> None:
    """Write a result dict (already fully JSON-serializable) to ``path``."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=indent, sort_keys=True)
        handle.write("\n")


def flatten_for_csv(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a result dict into one CSV-ready row.

    Scalar fields (``str``/``int``/``float``/``bool``/``None``) pass through
    unchanged. List/dict-valued fields (``interface_residues``,
    ``severe_clash_pairs``, ``active_sidechain_per_residue``,
    ``alignment_rotation``, ``delta_vs_previous_stage``, ...) are
    JSON-encoded into a single string cell so the row stays one value per
    column; their counts are already available as separate scalar fields
    (``interface_residue_count``, ``num_severe_clashes``, ...) where
    applicable.
    """
    row: Dict[str, Any] = {}
    for key, value in result.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            row[key] = value
        else:
            row[key] = json.dumps(value, sort_keys=True)
    return row


def append_to_csv(result: Mapping[str, Any], path: PathLike) -> None:
    """Append one evaluation's flattened result as a row to ``path``.

    Creates the file with a header on first write. If the file already
    exists, its header's column set is compared against the new row's
    columns; a mismatch raises rather than silently producing a ragged or
    misaligned CSV.

    Raises:
        ValueError: If ``path`` already exists with a different column set.
    """
    row = flatten_for_csv(result)
    csv_path = Path(path)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    if file_exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header and set(existing_header) != set(row):
            raise ValueError(
                f"CSV schema mismatch appending to {csv_path}: "
                f"existing columns {sorted(existing_header)} != new columns {sorted(row)}"
            )
        fieldnames = existing_header or sorted(row)
    else:
        fieldnames = sorted(row)
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_trajectory_to_csv(trajectory: Mapping[str, Any], path: PathLike) -> None:
    """Append every stage of one :func:`evaluate_trajectory` result as one CSV row each.

    Each row is that stage's full result dict (including ``"stage"`` and
    ``"stage_index"``), flattened exactly like :func:`flatten_for_csv`; all
    stages of one trajectory share one schema, so a schema mismatch across
    stages -- which should not happen since every stage is produced by the
    same :func:`evaluate_complex_metrics` call shape -- still raises via
    :func:`append_to_csv`.
    """
    for stage_result in trajectory["stages"]:
        append_to_csv(stage_result, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_stage_argument(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--stage expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--stage expects NAME=PATH, got {value!r}")
    return name, path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Reference structure (.cif or .pdb)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred", help="Single predicted/relaxed structure (.cif or .pdb)")
    mode.add_argument(
        "--stage", action="append", type=_parse_stage_argument, dest="stages",
        metavar="NAME=PATH",
        help="One pipeline stage (repeatable); switches to trajectory mode",
    )
    parser.add_argument("--receptor-chains", required=True, nargs="+", help="Antigen (receptor) chain IDs")
    parser.add_argument("--ligand-chains", required=True, nargs="+", help="Nanobody (ligand) chain IDs")
    parser.add_argument(
        "--active-residues", nargs="*", default=(),
        help='Residue ids ("CHAIN:SEQID[ICODE]") scored for symmetry-corrected '
             "side-chain RMSD and excluded from the alignment fit where on the receptor",
    )
    parser.add_argument("--fnat-cutoff", type=float, default=5.0)
    parser.add_argument("--interface-cutoff", type=float, default=10.0)
    parser.add_argument("--clash-cutoff", type=float, default=1.5)
    parser.add_argument("--peptide-bond-cutoff", type=float, default=1.9)
    parser.add_argument("--disulfide-cutoff", type=float, default=2.3)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="Write the result dict to this JSON path")
    parser.add_argument("--csv-out", default=None, help="Append the flattened result(s) to this CSV path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Evaluate one complex (or trajectory) from the command line and print/export the result."""

    args = _build_argument_parser().parse_args(argv)
    common = dict(
        receptor_chains=args.receptor_chains,
        ligand_chains=args.ligand_chains,
        active_residues=args.active_residues,
        fnat_cutoff=args.fnat_cutoff,
        interface_cutoff=args.interface_cutoff,
        clash_cutoff=args.clash_cutoff,
        peptide_bond_cutoff=args.peptide_bond_cutoff,
        disulfide_cutoff=args.disulfide_cutoff,
        model_index=args.model_index,
    )
    try:
        if args.stages:
            result: Dict[str, Any] = evaluate_trajectory(args.ref, dict(args.stages), **common)
        else:
            result = evaluate_complex_metrics(args.ref, args.pred, **common)
    except (ValueError, OSError) as exc:
        print(f"evaluate_complex_metrics: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        to_json(result, args.json_out)
    if args.csv_out:
        if args.stages:
            append_trajectory_to_csv(result, args.csv_out)
        else:
            append_to_csv(result, args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
