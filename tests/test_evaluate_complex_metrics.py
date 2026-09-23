"""Offline (no-gemmi-parsing) checks for evaluate_complex_metrics.

read_structure_atoms is monkeypatched to return hand-built fixtures (same
approach as this project's test_atomistic_evaluation.py), so these tests
exercise every pure-Python/numpy code path -- alignment, Fnat/iRMSD/LRMSD,
symmetry-corrected Active side-chain RMSD, disulfide/peptide-aware clash
exclusion, and multi-stage trajectory evaluation -- without needing the real
gemmi package or real structure files.
"""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import nanoqc.structure.evaluate_complex_metrics as m


def _vec(*xyz):
    a = np.array(xyz, dtype=np.float64)
    a.setflags(write=False)
    return a


# A backbone+CB template with an exact 1.33 A peptide-bond spacing when
# consecutive residues are offset by SPACING along x (matches the geometry
# already validated for the earlier evaluate_complex_dockq.py's offline tests).
_TEMPLATE = dict(N=(0., 0., 0.), CA=(1.46, .4, 0.), C=(2.0, -.6, 0.), O=(2.0, -.6, 1.8), CB=(1.46, 1.9, .3))
_SPACING = 3.33


def _backbone_chain(chain, start_seqid, count, name="ALA", base=(0., 0., 0.)):
    """count ALA residues in a straight peptide-bonded chain starting at base."""
    residues = {}
    for i in range(count):
        offset = np.array(base) + np.array([i * _SPACING, 0., 0.])
        atoms = {n: _vec(*(np.array(p) + offset)) for n, p in _TEMPLATE.items() if n != "CB" or name != "GLY"}
        residues[f"{chain}:{start_seqid + i}"] = dict(
            chain=chain, seqid=start_seqid + i, icode="", name=name, altloc="", atoms=atoms,
        )
    return residues


def _phe_ring(base):
    """A non-trivial (non-accidentally-symmetric) PHE ring anchored at CB=base."""
    base = np.array(base)
    cg = base + [0., 1.5, 0.]
    cd1 = cg + [0.9, 1.1, 0.4]
    cd2 = cg + [-0.7, 1.2, -0.5]
    ce1 = cd1 + [0.4, 1.3, 0.3]
    ce2 = cd2 + [-0.3, 1.3, -0.4]
    cz = cg + [0.1, 3.7, -0.05]
    return dict(CG=_vec(*cg), CD1=_vec(*cd1), CD2=_vec(*cd2), CE1=_vec(*ce1), CE2=_vec(*ce2), CZ=_vec(*cz))


def _make_complex(receptor_count=6, ligand_count=4, ligand_offset=(0., 5., 0.), phe_at=None):
    """A two-chain (receptor 'A', ligand 'H') all-ALA complex; optionally make one
    ligand residue a PHE with a ring, for symmetry-correction tests."""
    receptor = _backbone_chain("A", 1, receptor_count)
    ligand = _backbone_chain("H", 1, ligand_count, base=ligand_offset)
    if phe_at is not None:
        rid = f"H:{phe_at}"
        entry = ligand[rid]
        entry["name"] = "PHE"
        entry["atoms"] = {**entry["atoms"], **_phe_ring(entry["atoms"]["CB"])}
    return {**receptor, **ligand}


def _rotate(atoms, rotation=None, translation=(8., -4., 3.)):
    """Deep-copy atoms under a rigid transform (tests alignment/frame independence)."""
    if rotation is None:
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array(translation)
    out = {}
    for rid, entry in atoms.items():
        out[rid] = {**entry, "atoms": {n: _vec(*(xyz @ rotation + translation)) for n, xyz in entry["atoms"].items()}}
    return out


class EvaluateComplexMetricsTests(unittest.TestCase):
    def _evaluate(self, ref, pred, **kwargs):
        with patch("nanoqc.structure.evaluate_complex_metrics.read_structure_atoms", side_effect=[ref, pred]):
            return m.evaluate_complex_metrics(
                "ref.cif", "pred.cif",
                receptor_chains=["A"], ligand_chains=["H"], **kwargs,
            )

    def test_rigid_transform_invariance_and_perfect_scores(self):
        ref = _make_complex()
        pred = _rotate(ref)
        result = self._evaluate(ref, pred)
        self.assertAlmostEqual(result["fnat"], 1.0)
        self.assertLess(result["irmsd_angstrom"], 1e-8)
        self.assertLess(result["lrmsd_angstrom"], 1e-8)
        self.assertAlmostEqual(result["dockq_receptor_aligned_variant"], 1.0, places=6)
        self.assertEqual(result["dockq_category"], "High")
        self.assertEqual(result["num_severe_clashes"], 0)
        self.assertFalse(result["has_severe_clash"])
        self.assertIn("not directly comparable to published DockQ", result["dockq_definition"])

    def test_ligand_is_never_fit_only_transformed(self):
        # A ligand-only extra perturbation (not shared with the receptor) must
        # show up in LRMSD -- proof the alignment never fits the ligand away.
        ref = _make_complex()
        pred = _rotate(ref)
        pred["H:2"]["atoms"]["CA"] = _vec(*(pred["H:2"]["atoms"]["CA"] + np.array([4., 0., 0.])))
        result = self._evaluate(ref, pred)
        self.assertGreater(result["lrmsd_angstrom"], 0.9)
        # The receptor-only alignment fit itself must still be essentially exact.
        self.assertLess(result["alignment_rmsd_angstrom"], 1e-6)

    def test_active_residue_excluded_from_alignment_when_on_receptor(self):
        ref = _make_complex()
        pred = _rotate(ref)
        # Perturb one receptor residue's backbone; declaring it Active must
        # exclude it from the alignment fit so it doesn't drag the whole fit.
        pred["A:3"]["atoms"]["CA"] = _vec(*(pred["A:3"]["atoms"]["CA"] + np.array([5., 0., 0.])))
        with_active = self._evaluate(ref, pred, active_residues=["A:3"])
        without_active = self._evaluate(ref, pred)
        self.assertLess(with_active["alignment_rmsd_angstrom"], without_active["alignment_rmsd_angstrom"])
        self.assertLess(with_active["alignment_rmsd_angstrom"], 1e-6)
        self.assertEqual(with_active["alignment_residue_count"], 5)

    def test_phe_symmetric_ring_flip_is_corrected(self):
        ref = _make_complex(phe_at=2)
        pred = copy.deepcopy(ref)
        entry = pred["H:2"]["atoms"]
        entry["CD1"], entry["CD2"] = entry["CD2"], entry["CD1"]
        entry["CE1"], entry["CE2"] = entry["CE2"], entry["CE1"]
        result = self._evaluate(ref, pred, active_residues=["H:2"])
        self.assertIsNotNone(result["active_sidechain_rmsd_angstrom"])
        self.assertLess(result["active_sidechain_rmsd_angstrom"], 1e-8)
        detail = result["active_sidechain_per_residue"][0]
        self.assertEqual(detail["residue_id"], "H:2")
        self.assertTrue(detail["symmetry_corrected"])
        self.assertLess(detail["sidechain_rmsd_angstrom"], 1e-8)

    def test_phe_partial_swap_is_not_spuriously_corrected(self):
        # Swapping only CD1/CD2 without the paired CE1/CE2 swap is not a
        # physically valid ring flip; the minimum-over-2-candidates search
        # must NOT hide this behind the full-swap candidate.
        ref = _make_complex(phe_at=2)
        pred = copy.deepcopy(ref)
        entry = pred["H:2"]["atoms"]
        entry["CD1"], entry["CD2"] = entry["CD2"], entry["CD1"]
        result = self._evaluate(ref, pred, active_residues=["H:2"])
        self.assertGreater(result["active_sidechain_rmsd_angstrom"], 0.3)

    def test_unswapped_active_residue_scores_true_error(self):
        ref = _make_complex()
        pred = copy.deepcopy(ref)
        pred["H:2"]["atoms"]["CB"] = _vec(*(pred["H:2"]["atoms"]["CB"] + np.array([1., 0., 0.])))
        result = self._evaluate(ref, pred, active_residues=["H:2"])
        self.assertAlmostEqual(result["active_sidechain_rmsd_angstrom"], 1.0, places=6)
        self.assertFalse(result["active_sidechain_per_residue"][0]["symmetry_corrected"])

    def test_gly_active_residue_contributes_no_sidechain_atoms(self):
        ref = _make_complex()
        ref["A:1"]["name"] = "GLY"
        ref["A:1"]["atoms"] = {k: v for k, v in ref["A:1"]["atoms"].items() if k != "CB"}
        pred = copy.deepcopy(ref)
        result = self._evaluate(ref, pred, active_residues=["A:1"])
        self.assertIsNone(result["active_sidechain_rmsd_angstrom"])
        self.assertEqual(result["active_sidechain_per_residue"][0]["sidechain_atom_count"], 0)

    def test_peptide_bond_and_disulfide_pairs_excluded_from_clashes(self):
        ref = _make_complex(receptor_count=3, ligand_count=2)
        # Two CYS residues forming a genuine disulfide at 2.05 A.
        ref["A:1"]["name"] = "CYS"
        ref["A:1"]["atoms"]["SG"] = _vec(10., 10., 10.)
        ref["A:3"]["name"] = "CYS"
        ref["A:3"]["atoms"]["SG"] = _vec(10., 10., 12.0499)  # 2.0499 A away < 2.3 A cutoff
        pred = copy.deepcopy(ref)
        result = self._evaluate(ref, pred)
        # The peptide C(1)-N(2)/C(2)-N(3) bonds (by construction, exactly 1.33 A)
        # and the SG-SG disulfide must both be excluded -- zero clashes reported.
        self.assertEqual(result["num_severe_clashes"], 0)
        self.assertIn("disulfide", result["clash_definition"])

    def test_genuine_clash_is_detected(self):
        ref = _make_complex(receptor_count=3, ligand_count=2)
        pred = copy.deepcopy(ref)
        # Force two non-bonded atoms (receptor res 1 O, receptor res 3 N) into
        # a genuine steric clash that is not a peptide or disulfide exclusion.
        pred["A:3"]["atoms"]["N"] = _vec(*(pred["A:1"]["atoms"]["O"] + np.array([0.1, 0., 0.])))
        result = self._evaluate(ref, pred)
        self.assertTrue(result["has_severe_clash"])
        self.assertGreaterEqual(result["num_severe_clashes"], 1)

    def test_undefined_dockq_when_no_native_contacts(self):
        ref = _make_complex(ligand_offset=(0., 500., 0.))  # far away: zero native contacts
        pred = _rotate(ref)
        result = self._evaluate(ref, pred)
        self.assertIsNone(result["fnat"])
        self.assertIsNone(result["dockq_receptor_aligned_variant"])
        self.assertIsNone(result["dockq_category"])

    def test_invalid_arguments_rejected(self):
        ref = _make_complex()
        pred = copy.deepcopy(ref)
        with patch("nanoqc.structure.evaluate_complex_metrics.read_structure_atoms", side_effect=lambda *a, **k: copy.deepcopy(ref)):
            with self.assertRaises(ValueError):
                m.evaluate_complex_metrics("r", "p", receptor_chains=["A"], ligand_chains=["A"])
            with self.assertRaises(ValueError):
                m.evaluate_complex_metrics("r", "p", receptor_chains=[], ligand_chains=["H"])
            with self.assertRaises(ValueError):
                m.evaluate_complex_metrics("r", "p", receptor_chains=["A"], ligand_chains=["H"], active_residues=["Z:99"])
            with self.assertRaises(ValueError):
                m.evaluate_complex_metrics("r", "p", receptor_chains=["A"], ligand_chains=["H"], fnat_cutoff=-1.0)
            # Excluding too many receptor residues from alignment (all but 2) must fail.
            with self.assertRaises(ValueError):
                m.evaluate_complex_metrics(
                    "r", "p", receptor_chains=["A"], ligand_chains=["H"],
                    active_residues=[f"A:{i}" for i in range(1, 6)],
                )

    def test_trajectory_multi_stage_and_deltas(self):
        ref = _make_complex()
        perturbed = copy.deepcopy(ref)
        perturbed["H:2"]["atoms"]["CA"] = _vec(*(perturbed["H:2"]["atoms"]["CA"] + np.array([3., 0., 0.])))
        relaxed = copy.deepcopy(ref)
        relaxed["H:2"]["atoms"]["CA"] = _vec(*(relaxed["H:2"]["atoms"]["CA"] + np.array([1., 0., 0.])))
        final = copy.deepcopy(ref)
        stage_paths = {"perturbed_input": "p.cif", "stage1_relaxed": "s1.cif", "stage2_relaxed": "s2.cif"}
        with patch(
            "nanoqc.structure.evaluate_complex_metrics.read_structure_atoms",
            side_effect=[ref, perturbed, ref, relaxed, ref, final],
        ):
            trajectory = m.evaluate_trajectory(
                "ref.cif", stage_paths, receptor_chains=["A"], ligand_chains=["H"],
            )
        self.assertEqual(trajectory["stage_order"], ["perturbed_input", "stage1_relaxed", "stage2_relaxed"])
        self.assertEqual([s["stage"] for s in trajectory["stages"]], trajectory["stage_order"])
        self.assertEqual([s["stage_index"] for s in trajectory["stages"]], [0, 1, 2])
        self.assertEqual(trajectory["stages"][0]["delta_vs_previous_stage"], {})
        # LRMSD should strictly improve (decrease) perturbed -> stage1 -> stage2 (final == ref).
        lrmsds = [s["lrmsd_angstrom"] for s in trajectory["stages"]]
        self.assertGreater(lrmsds[0], lrmsds[1])
        self.assertGreater(lrmsds[1], lrmsds[2])
        self.assertLess(lrmsds[2], 1e-8)
        delta_1 = trajectory["stages"][1]["delta_vs_previous_stage"]
        self.assertAlmostEqual(delta_1["lrmsd_angstrom"], lrmsds[1] - lrmsds[0])
        self.assertLess(delta_1["lrmsd_angstrom"], 0.0)  # improvement = negative delta

    def test_trajectory_requires_at_least_one_stage(self):
        with self.assertRaises(ValueError):
            m.evaluate_trajectory("ref.cif", {}, receptor_chains=["A"], ligand_chains=["H"])

    def test_trajectory_error_names_failing_stage(self):
        ref = _make_complex()
        broken = copy.deepcopy(ref)
        del broken["H:2"]  # missing residue -> ValueError from evaluate_complex_metrics
        with patch("nanoqc.structure.evaluate_complex_metrics.read_structure_atoms", side_effect=[ref, broken]):
            with self.assertRaisesRegex(ValueError, "bad_stage"):
                m.evaluate_trajectory(
                    "ref.cif", {"bad_stage": "x.cif"}, receptor_chains=["A"], ligand_chains=["H"],
                )

    def test_json_and_csv_export_round_trip(self):
        ref = _make_complex()
        pred = _rotate(ref)
        result = self._evaluate(ref, pred)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "out.json"
            csv_path = Path(tmp) / "out.csv"
            m.to_json(result, json_path)
            reloaded = json.loads(json_path.read_text())
            self.assertAlmostEqual(reloaded["dockq_receptor_aligned_variant"], result["dockq_receptor_aligned_variant"])
            m.append_to_csv(result, csv_path)
            m.append_to_csv(result, csv_path)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("dockq_receptor_aligned_variant", rows[0])

    def test_trajectory_csv_export_one_row_per_stage(self):
        ref = _make_complex()
        stage_paths = {"perturbed_input": "p.cif", "relax_only": "r.cif"}
        with patch("nanoqc.structure.evaluate_complex_metrics.read_structure_atoms", side_effect=[ref, ref, ref, ref]):
            trajectory = m.evaluate_trajectory(
                "ref.cif", stage_paths, receptor_chains=["A"], ligand_chains=["H"],
            )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trajectory.csv"
            m.append_trajectory_to_csv(trajectory, csv_path)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["stage"] for r in rows], ["perturbed_input", "relax_only"])


if __name__ == "__main__":
    unittest.main()
