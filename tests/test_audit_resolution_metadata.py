"""Entry resolution from curation metadata; pipeline staging is never audited."""
from __future__ import annotations

import pytest

from nanoqc.data import audit_all_datasets as audit

SNAC_HEADER = "Name,Resolution,PDB_ID,Chain_VH,Chain_VH_old_id,Chain_Ag\n"


def test_parse_resolution_reads_the_worst_number_or_none():
    assert audit.parse_resolution("2.5") == 2.5
    assert audit.parse_resolution("2.5, 2.7") == 2.7
    for missing in ("Resolution is Missing", "NOT", "", None, "0"):
        assert audit.parse_resolution(missing) is None


def test_snac_and_sabdab_metadata_supply_entry_resolution(tmp_path):
    curated = tmp_path / "SNAC-DataBase" / "curated_structures"
    curated.mkdir(parents=True)
    (curated / "nb_complexes_curation_summary.csv").write_text(
        SNAC_HEADER + "9DVQ-ASU1-VHH_G-Ag_A,2.1,9DVQ,H,G,['A']\n1BEC-ASU0-VHH_A,Resolution is Missing,1BEC,H,A,[]\n")
    (tmp_path / "sabdab_summary_all.tsv").write_text(
        "pdb\tHchain\tresolution\n9dvq\tH\t3.9\n7abc\tB\t2.8, 2.8\n1bec\tA\tNOT\n")
    audit.load_annotations(tmp_path)
    try:
        # SNAC is read first and wins; SAbDab fills entries SNAC lacks.
        assert audit.RESOLUTION_BY_PDB["9DVQ"] == (2.1, "snac_curation_summary")
        assert audit.RESOLUTION_BY_PDB["7ABC"] == (2.8, "sabdab_summary")
        assert "1BEC" not in audit.RESOLUTION_BY_PDB
    finally:
        audit.load_annotations(tmp_path / "none")


def test_external_vhh_staging_is_not_a_study_subset(tmp_path):
    (tmp_path / "external_vhh" / "prep" / "foldseek" / "antigen_structures").mkdir(parents=True)
    (tmp_path / "external_vhh" / "prep" / "foldseek" / "antigen_structures" / "1abc.cif").write_text("data_x\n")
    (tmp_path / "train_rcsb").mkdir()
    (tmp_path / "train_rcsb" / "1abc.pdb").write_text("END\n")
    tasks, _, _ = audit.discover(tmp_path)
    assert [t["subset"] for t in tasks] == ["train_rcsb"]


def test_the_worst_resolution_of_a_multi_entry_field_decides(tmp_path):
    """A field may list one value per entry [R33]; the gate is an upper bound."""
    curated = tmp_path / "SNAC-DataBase" / "curated_structures"
    curated.mkdir(parents=True)
    (curated / "nb_complexes_curation_summary.csv").write_text(
        SNAC_HEADER + "A,\"2.5, 3.4\",7AAA,H,A,['B']\n")
    audit.load_annotations(tmp_path)
    try:
        assert audit.RESOLUTION_BY_PDB["7AAA"][0] == 3.4  # not 2.5: would pass a 3.0 gate
    finally:
        audit.load_annotations(tmp_path / "none")


def test_staging_metadata_never_decides_the_gate_and_used_files_are_hashed(tmp_path):
    (tmp_path / "external_vhh").mkdir()
    (tmp_path / "external_vhh" / "sabdab_summary_sd_h.tsv").write_text(
        "pdb\tresolution\n8bbb\t1.5\n")
    (tmp_path / "sabdab_summary_all.tsv").write_text("pdb\tresolution\n9ccc\t2.0\n")
    audit.load_annotations(tmp_path)
    try:
        assert "8BBB" not in audit.RESOLUTION_BY_PDB  # staging is not study input
        assert audit.RESOLUTION_BY_PDB["9CCC"] == (2.0, "sabdab_summary")
        used = audit.RESOLUTION_SOURCE_FILES
        assert [f["path"] for f in used] == ["sabdab_summary_all.tsv"]
        assert len(used[0]["sha256"]) == 64
    finally:
        audit.load_annotations(tmp_path / "none")


ATOMS = """loop_
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
ATOM 1 N N . ALA A 1 1 0 0 0 1 10 1 A 1
ATOM 2 C CA . ALA A 1 1 1 0 0 1 10 1 A 1
ATOM 3 C C . ALA A 1 1 2 0 0 1 10 1 A 1
ATOM 4 O O . ALA A 1 1 3 0 0 1 10 1 A 1
"""


def test_a_downloaded_assembly_file_is_the_assembly_and_is_not_rebuilt(tmp_path):
    """RCSB serves <entry>_assembly<N>: no pdbx_struct_assembly, already assembled."""
    from nanoqc.data.audit_all_datasets import prebuilt_assembly, read_structure

    assert prebuilt_assembly("10bt_assembly1.cif") == "1"
    assert prebuilt_assembly("1abc-assembly2.cif.gz") == "2"
    assert prebuilt_assembly("1abc.pdb1") == "1"
    for plain in ("1abc.cif", "pdb_000010zo_sabdab.cif", "9DVQ-ASU1-VHH_G-Ag_A.pdb"):
        assert prebuilt_assembly(plain) is None

    path = tmp_path / "10bt_assembly1.cif"
    path.write_text("data_10bt\n_entry.id 10bt\n" + ATOMS)
    task = dict(path=str(path), member="", subset="train_rcsb", id=path.name)
    structure, _ = read_structure(task)
    assert dict(structure.info)[audit.STRUCTURE_SOURCE_KEY] == "prebuilt_assembly_file:assembly1"
    assert len(structure[0]) == 1  # coordinates untouched, no second transform


def test_an_asymmetric_unit_without_assembly_annotation_still_fails_closed(tmp_path):
    from nanoqc.data.audit_all_datasets import read_structure

    path = tmp_path / "10bt.cif"
    path.write_text("data_10bt\n_entry.id 10bt\n" + ATOMS)
    task = dict(path=str(path), member="", subset="train_rcsb", id=path.name)
    with pytest.raises(ValueError, match="no biological assembly annotation"):
        read_structure(task)
