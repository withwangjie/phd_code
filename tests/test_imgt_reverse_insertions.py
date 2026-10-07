"""IMGT numbers CDR3 insertions at 112 in reverse (112B, 112A, 112): chain
order is not (seqid, insertion code) order, so neighbours come from bonds."""
import numpy as np

from nanoqc.qubo.atomistic_structure import _backbone_phi_psi
from nanoqc.structure.side_chain_metrics import compute_bonded_exclusions

ORDER = ["111", "112B", "112A", "112", "113"]          # file / chain order


def _residues():
    residues = {}
    for k, label in enumerate(ORDER):
        x = 3.8 * k
        residues[f"H:{label}"] = dict(
            chain="H", seqid=int(label[:3]), icode=label[3:], name="GLY",
            atoms={"N": np.array([x, 0.0, 0.0]), "CA": np.array([x + 1.46, 0.4, 0.0]),
                   "C": np.array([x + 2.47, 0.0, 0.3])})
    return residues


def test_phi_psi_uses_the_bonded_neighbours_of_a_reverse_insertion() -> None:
    residues = _residues()
    phi, psi = _backbone_phi_psi(residues, "H:112A")
    atoms = {k: v["atoms"] for k, v in residues.items()}
    from nanoqc.qubo.atomistic_structure import _torsion_angle_degrees
    assert np.isclose(phi, _torsion_angle_degrees(atoms["H:112B"]["C"], atoms["H:112A"]["N"],
                                                  atoms["H:112A"]["CA"], atoms["H:112A"]["C"]))
    assert np.isclose(psi, _torsion_angle_degrees(atoms["H:112A"]["N"], atoms["H:112A"]["CA"],
                                                  atoms["H:112A"]["C"], atoms["H:112"]["N"]))


def test_a_real_chain_break_still_fails_closed() -> None:
    residues = _residues()
    residues["H:112"]["atoms"] = {k: v + np.array([0.0, 9.0, 0.0])
                                  for k, v in residues["H:112"]["atoms"].items()}
    try:
        _backbone_phi_psi(residues, "H:112A")
    except ValueError as exc:
        assert "peptide-bonded neighbours" in str(exc)
    else:
        raise AssertionError("chain break was not detected")


def test_peptide_bonds_of_reverse_insertions_are_excluded_from_clashes() -> None:
    residues = _residues()
    excluded = compute_bonded_exclusions(residues, list(residues))
    for left, right in zip(ORDER, ORDER[1:]):
        assert frozenset(((f"H:{left}", "C"), (f"H:{right}", "N"))) in excluded
