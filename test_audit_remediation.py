"""End-to-end pytest regression suite for the audit-remediation patches.

Verifies, against the actual shipped modules (not reimplementations of
their logic):

* Patch 1 (evaluate_complex_metrics.py -- verify_full_chain_heavy_atom_
  completeness): full-chain, residue-by-residue heavy-atom completeness is
  enforced across every evaluated chain, not only active/interface atoms --
  deleting an off-interface, non-Active heavy atom from the prediction
  raises AtomCompletenessError immediately instead of being silently
  absorbed into a lower score.
* Patch 2 (qaoa_interface_sampler.py -- optimize_robust): total restart
  collapse (every restart's COBYLA call raising) is reported honestly via
  optimizer_success=False and termination_reason=="all_restarts_failed",
  never silently upgraded to a misleading "the run basically worked"
  classification such as "mixed_or_incomplete_convergence".
* Patch 3 (evaluate_complex_metrics.py -- evaluate_trajectory): multi-stage
  pipeline evaluation populates dockq_receptor_aligned_variant and
  delta_vs_previous_stage at every stage, with a schema-consistent
  (present, never omitted -- empty for the first stage) delta dict.

Structures are synthetic, hand-built fixtures -- the same
read_structure_atoms-monkeypatching approach this project's existing
test_evaluate_complex_metrics.py already uses -- so these tests need no
real PDB/mmCIF files and no network access; only the real
evaluate_complex_metrics/qaoa_interface_sampler modules and their existing
gemmi/pennylane/scipy/numpy dependencies (already required by this
project's other offline test suites).

Run with: pytest test_audit_remediation.py -v
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple
from unittest.mock import patch

import numpy as np
import pytest

import evaluate_complex_metrics as ecm
import qaoa_interface_sampler as qis


# ---------------------------------------------------------------------------
# Shared synthetic-structure fixtures
# ---------------------------------------------------------------------------

def _vec(*xyz: float) -> np.ndarray:
    array = np.array(xyz, dtype=np.float64)
    array.setflags(write=False)
    return array


# Backbone+CB template with an exact 1.33 A peptide-bond spacing when
# consecutive residues are offset by _SPACING along x -- matches this
# project's existing offline-test geometry convention (see
# test_evaluate_complex_metrics.py's _TEMPLATE/_backbone_chain).
_TEMPLATE: Dict[str, Tuple[float, float, float]] = dict(
    N=(0.0, 0.0, 0.0), CA=(1.46, 0.4, 0.0), C=(2.0, -0.6, 0.0), O=(2.0, -0.6, 1.8), CB=(1.46, 1.9, 0.3),
)
_SPACING = 3.33


def _backbone_chain(
    chain: str, start_seqid: int, count: int, *, name: str = "ALA", base: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Dict[str, Dict[str, Any]]:
    """``count`` ALA residues in a straight peptide-bonded chain starting at ``base``."""
    residues: Dict[str, Dict[str, Any]] = {}
    for i in range(count):
        offset = np.array(base) + np.array([i * _SPACING, 0.0, 0.0])
        atoms = {n: _vec(*(np.array(p) + offset)) for n, p in _TEMPLATE.items() if n != "CB" or name != "GLY"}
        residues[f"{chain}:{start_seqid + i}"] = dict(
            chain=chain, seqid=start_seqid + i, icode="", name=name, altloc="", atoms=atoms,
        )
    return residues


def _make_complex(
    receptor_count: int = 20, ligand_count: int = 4,
    ligand_offset: Tuple[float, float, float] = (0.0, 5.0, 0.0),
) -> Dict[str, Dict[str, Any]]:
    """A two-chain (receptor 'A', ligand 'H') all-ALA synthetic complex."""
    receptor = _backbone_chain("A", 1, receptor_count)
    ligand = _backbone_chain("H", 1, ligand_count, base=ligand_offset)
    return {**receptor, **ligand}


# ---------------------------------------------------------------------------
# Patch 1: full-chain heavy-atom completeness (evaluate_complex_metrics.py)
# ---------------------------------------------------------------------------

def test_atom_completeness_rejection() -> None:
    """Deleting a non-interface, non-Active heavy atom (receptor A:20/CB)
    from the predicted structure must raise -- not be silently ignored
    because A:20 is neither an Active residue nor near the interface.

    This is exactly the "off-interface heavy-atom bypass" the audit
    identified: prior to Patch 1, only backbone/interface/Active atoms were
    checked, so this deletion would previously have been scored past
    silently.
    """
    ref = _make_complex(receptor_count=20, ligand_count=4, ligand_offset=(0.0, 5.0, 0.0))
    pred = copy.deepcopy(ref)
    del pred["A:20"]["atoms"]["CB"]

    with patch("evaluate_complex_metrics.read_structure_atoms", side_effect=[ref, pred]):
        with pytest.raises(ecm.AtomCompletenessError) as excinfo:
            ecm.evaluate_complex_metrics(
                "ref.cif", "pred.cif", receptor_chains=["A"], ligand_chains=["H"],
            )

    message = str(excinfo.value)
    assert "A:20" in message
    assert "CB" in message
    # AtomCompletenessError subclasses ValueError, so existing
    # `except ValueError` call sites elsewhere in the pipeline keep working
    # unchanged (backward compatibility), while a caller that wants to
    # distinguish this specific failure mode can catch it directly.
    assert isinstance(excinfo.value, ValueError)


def test_atom_completeness_accepts_a_complete_structure() -> None:
    """Sanity check: an unmodified predicted structure passes the same
    check -- proves the rejection above is caused by the deletion, not a
    fixture defect."""
    ref = _make_complex(receptor_count=20, ligand_count=4, ligand_offset=(0.0, 5.0, 0.0))
    pred = copy.deepcopy(ref)

    with patch("evaluate_complex_metrics.read_structure_atoms", side_effect=[ref, pred]):
        result = ecm.evaluate_complex_metrics(
            "ref.cif", "pred.cif", receptor_chains=["A"], ligand_chains=["H"],
        )

    assert result["dockq_receptor_aligned_variant"] is not None
    assert result["lrmsd_angstrom"] < 1e-8


# ---------------------------------------------------------------------------
# Patch 2: honest all-restarts-failed handling (qaoa_interface_sampler.py)
# ---------------------------------------------------------------------------

def test_all_restarts_failed_handling() -> None:
    """Stub the classical optimizer to raise on every restart; optimize_robust
    must not silently fall back to the uniform baseline as if the run
    succeeded -- it must report total algorithmic collapse honestly via
    optimizer_success=False and termination_reason=="all_restarts_failed",
    distinct from ordinary "max_evaluations_reached" budget exhaustion.
    """
    sampler = qis.XYMixerQAOASampler(
        [0.0, 2.0, 0.0, 5.0], np.zeros((4, 4)), {0: [0, 1], 1: [2, 3]},
        p=2, simulation_mode="subspace", seed=42,
    )

    with patch("qaoa_interface_sampler.minimize", side_effect=RuntimeError("stubbed restart failure")):
        result = sampler.optimize_robust(max_evals=25, restarts=4, eval_shots=50)

    assert result.optimizer_success is False
    assert result.termination_reason == "all_restarts_failed"
    assert result.success is False  # legacy alias must agree with optimizer_success
    assert result.raw_result is not None
    assert result.raw_result["restart_count"] == 4
    assert len(result.raw_result["restart_exception_messages"]) == 4
    assert all("stubbed restart failure" in msg for msg in result.raw_result["restart_exception_messages"])

    # Every restart record individually reflects the same failure mode.
    assert len(result.restart_records) == 4
    assert all(r["termination_reason"] == "restart_raised_exception" for r in result.restart_records)

    # A well-formed (if degenerate -- the cost-free uniform-state baseline)
    # result is still returned, never a crash, so a caller can inspect it.
    assert result.gammas.shape == (2,)
    assert result.betas.shape == (2,)
    assert np.isfinite(result.energy)
    np.testing.assert_array_equal(result.gamma_best, result.gammas)
    np.testing.assert_array_equal(result.beta_best, result.betas)


def test_all_restarts_failed_is_distinct_from_budget_exhaustion() -> None:
    """termination_reason=="all_restarts_failed" must never be confused with
    ordinary budget exhaustion (unstubbed optimize_robust under a tight
    budget still runs real restarts and reports max_evaluations_reached)."""
    sampler = qis.XYMixerQAOASampler(
        [0.0, 2.0, 0.0, 5.0], np.zeros((4, 4)), {0: [0, 1], 1: [2, 3]},
        p=2, simulation_mode="subspace", seed=17,
    )
    result = sampler.optimize_robust(max_evals=25, restarts=4, eval_shots=30)
    assert result.termination_reason != "all_restarts_failed"
    assert result.raw_result is None


# ---------------------------------------------------------------------------
# Patch 3: pipeline integration -- multi-stage trajectory evaluation
# ---------------------------------------------------------------------------

def test_trajectory_integration() -> None:
    """A five-stage pipeline sequence (this project's canonical checkpoints:
    perturbed_input, relax_only, discrete_picked, stage1_relaxed,
    stage2_relaxed) must populate dockq_receptor_aligned_variant and
    delta_vs_previous_stage at every stage, with a schema-consistent
    (present, never omitted) delta dict even for the first stage.
    """
    ref = _make_complex(receptor_count=10, ligand_count=4, ligand_offset=(0.0, 5.0, 0.0))

    def _nudge(structure: Dict[str, Dict[str, Any]], magnitude: float) -> Dict[str, Dict[str, Any]]:
        out = copy.deepcopy(structure)
        out["H:2"]["atoms"]["CA"] = _vec(*(out["H:2"]["atoms"]["CA"] + np.array([magnitude, 0.0, 0.0])))
        return out

    stage_names = ["perturbed_input", "relax_only", "discrete_picked", "stage1_relaxed", "stage2_relaxed"]
    magnitudes = [4.0, 3.0, 2.0, 1.0, 0.0]  # monotonically improving toward the exact reference
    stage_paths = {name: f"{name}.cif" for name in stage_names}
    stage_structures = [_nudge(ref, m) for m in magnitudes]

    # evaluate_complex_metrics reads (reference, prediction) once per stage.
    read_sequence = []
    for structure in stage_structures:
        read_sequence.extend([ref, structure])

    with patch("evaluate_complex_metrics.read_structure_atoms", side_effect=read_sequence):
        trajectory = ecm.evaluate_trajectory(
            "ref.cif", stage_paths, receptor_chains=["A"], ligand_chains=["H"],
        )

    assert trajectory["stage_order"] == stage_names
    assert [s["stage"] for s in trajectory["stages"]] == stage_names
    assert [s["stage_index"] for s in trajectory["stages"]] == list(range(5))
    assert len(trajectory["stages"]) == 5

    for stage in trajectory["stages"]:
        assert "dockq_receptor_aligned_variant" in stage
        assert "delta_vs_previous_stage" in stage
        assert isinstance(stage["delta_vs_previous_stage"], dict)  # present at every stage, never omitted

    # Schema-consistent: the first stage has no previous stage to compare
    # against, so its delta dict is empty (present, not missing/None) --
    # this matters for append_trajectory_to_csv, where every stage of one
    # trajectory must append to the same CSV columns.
    assert trajectory["stages"][0]["delta_vs_previous_stage"] == {}

    # The last stage exactly matches the reference: dockq must reach a
    # well-defined, non-None value there (Fnat/iRMSD/LRMSD all defined).
    assert trajectory["stages"][-1]["dockq_receptor_aligned_variant"] is not None
    assert trajectory["stages"][-1]["lrmsd_angstrom"] < 1e-8

    # LRMSD strictly improves (decreases) across the monotonically-nudged
    # sequence, and delta_vs_previous_stage must reflect that sign.
    lrmsds = [s["lrmsd_angstrom"] for s in trajectory["stages"]]
    for later, earlier in zip(lrmsds[1:], lrmsds[:-1]):
        assert later <= earlier + 1e-9
    for stage in trajectory["stages"][1:]:
        if "lrmsd_angstrom" in stage["delta_vs_previous_stage"]:
            assert stage["delta_vs_previous_stage"]["lrmsd_angstrom"] <= 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
