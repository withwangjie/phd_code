from __future__ import annotations

import json
import sys

import numpy as np

from nanoqc.data import audit_all_datasets as audit
from nanoqc.pipeline.run_full_experiment import Orchestrator


def _chain(name, xyz, owners, atom_names):
    return {
        "name": name, "xyz": np.asarray(xyz, dtype=float),
        "owners": np.asarray(owners), "atom_names": atom_names,
        "residue_seqids": ["1", "2"], "residue_names": ["ALA", "ALA"],
    }


def test_absolute_floor_ignores_same_residue_bonds_and_counts_other_residues():
    left = _chain("H", [[0, 0, 0], [0.1, 0, 0], [5, 0, 0]],
                  [0, 0, 1], ["N", "CA", "N"])
    right = _chain("A", [[0.6, 0, 0]], [0], ["N"])
    result = audit.interresidue_heavy_overlap([left, right], 1.0)
    assert result["count"] == 2
    assert result["minimum_distance_angstrom"] == 0.5
    assert result["closest_pair"] == ["H:1:ALA:CA", "A:1:ALA:N"]
    assert audit.interresidue_heavy_overlap([left], 1.0)["count"] == 0


def test_formal_admission_rejects_a_nonphysical_quality_result():
    row = dict(subset="snac_db", valid=True, missing_residues=0,
               interface_status="pass", max_contact_residues=20,
               vhh_status="pass", structure_quality_status="fail",
               structure_quality_reasons=["nonphysical_interresidue_heavy_overlap"],
               interresidue_heavy_overlap_count=1)
    assert not audit.formal_row_eligible(row)


def test_structure_audit_records_the_distance_and_quality_exclusion(tmp_path, monkeypatch):
    header = """data_1abc
_entry.id 1abc
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""
    atoms=[]
    for chain, offset, entity in (("H", 0.0, 1), ("A", 0.4, 2)):
        for atom_name, element, x in (("N", "N", 0), ("CA", "C", 1),
                                      ("C", "C", 2), ("O", "O", 3),
                                      ("CB", "C", 1.5)):
            atoms.append(f"ATOM {len(atoms)+1} {element} {atom_name} . ALA {chain} "
                         f"{entity} 1 {x+offset:.3f} 0 0 1 10 1 {chain} 1")
    path = tmp_path / "1abc.cif"
    path.write_text(header + "\n".join(atoms) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "nano_features", lambda task, chains, pdb:
                        (dict(vhh_status="pass"), lambda a, b: {a, b} == {"H", "A"}))
    audit.RESOLUTION_BY_PDB["1ABC"] = (2.0, "test")
    try:
        row = audit.audit(dict(path=str(path), member="", subset="snac_db", id="1abc"))
    finally:
        audit.RESOLUTION_BY_PDB.pop("1ABC", None)
    assert row["valid"], row["error"]
    assert row["interresidue_heavy_overlap_count"] > 0
    assert row["interresidue_heavy_min_distance_angstrom"] < 1.0
    assert row["structure_quality_status"] == "fail"
    assert "nonphysical_interresidue_heavy_overlap" in row["structure_quality_reasons"]

    output = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit_all_datasets", "--data", str(tmp_path),
                                   "--out", str(output), "--workers", "1",
                                   "--min-interresidue-heavy-distance", "1.0"])
    audit.main()
    inventory = json.loads((output / "data_audit_inventory.json").read_text())
    counts = inventory["interresidue_heavy_overlap_audit"]
    assert counts["evaluated_rows"] == 1
    assert counts["excluded_rows"] == 1
    assert counts["overlap_atom_pairs"] > 0
    assert inventory["structure_quality_protocol"]["min_interresidue_heavy_distance_angstrom"] == 1.0

    class Harness:
        run_dir = tmp_path
        config = {"data_audit": {"min_interresidue_heavy_distance_angstrom": 1.0}}
        _artifacts_present = staticmethod(Orchestrator._artifacts_present)

    ok, detail = Orchestrator._validate_completed_stage_artifacts(
        Harness(), "data_audit", require_results_manifest=False
    )
    assert ok, detail


def test_db55_pairs_rows_are_identical_with_and_without_a_pool(monkeypatch):
    """The shared pool only moves the contact computation off the parent (A36)."""
    import concurrent.futures
    from nanoqc.data import audit_all_datasets as audit_module

    tasks = [dict(subset="test_db55", path=f"/d/{name}") for name in
             ("1abc_r_b.pdb", "1abc_l_b.pdb", "2def_r_b.pdb", "3ghi_r_b.pdb", "3ghi_l_b.pdb")]
    counts = {"/d/1abc_r_b.pdb": (9, 3), "/d/3ghi_r_b.pdb": (20, 14)}

    def fake_counts(pair):
        receptor, _partner = pair
        if receptor["path"] == "/d/3ghi_r_b.pdb":
            raise ValueError("empty receptor/ligand")
        return counts[receptor["path"]]

    monkeypatch.setattr(audit_module, "_db55_contact_counts", fake_counts)
    serial = audit_module.db55_pairs(tasks)

    class InlinePool:
        def map(self, function, items):
            return [function(item) for item in items]

    assert audit_module.db55_pairs(tasks, InlinePool()) == serial
    assert [row["id"] for row in serial] == ["1abc", "2def", "3ghi"]
    # Weak/failed/missing-partner outcomes are unchanged by the pool.
    by_id = {row["id"]: row for row in serial}
    assert by_id["1abc"]["interface_status"] == "weak" and by_id["1abc"]["contact_residues"] == 12
    assert by_id["2def"]["error"] == "missing bound ligand" and not by_id["2def"]["valid"]
    assert by_id["3ghi"]["error"] == "empty receptor/ligand" and not by_id["3ghi"]["valid"]
    # A pool whose results arrive as a lazy iterator is consumed in pair order.
    assert audit_module.db55_pairs(tasks, concurrent.futures.ThreadPoolExecutor(2)) == serial
