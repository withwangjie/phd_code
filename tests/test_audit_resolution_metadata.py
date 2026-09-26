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


def test_an_rcsb_entry_resolution_table_feeds_the_gate(tmp_path):
    """A15: files with no resolution record get it from a declared table."""
    (tmp_path / "rcsb_non_redundant_dataset").mkdir()
    (tmp_path / "rcsb_non_redundant_dataset" / "rcsb_entry_resolution.tsv").write_text(
        "pdb\tresolution\tmethod\n10BT\t2.1\tX-RAY DIFFRACTION\n10CY\t\tSOLUTION NMR\n")
    audit.load_annotations(tmp_path)
    try:
        assert audit.RESOLUTION_BY_PDB["10BT"] == (2.1, "rcsb_entry_resolution")
        assert "10CY" not in audit.RESOLUTION_BY_PDB  # NMR: no resolution, still excluded
        assert [f["kind"] for f in audit.RESOLUTION_SOURCE_FILES] == ["rcsb_entry_resolution"]
    finally:
        audit.load_annotations(tmp_path / "none")


def test_the_fetcher_reads_ids_and_the_worst_resolution():
    from nanoqc.data import fetch_entry_resolution as fetch

    assert fetch.entry_ids(["10bt", "10bt_assembly1.cif", " 6VXX ", "junk"]) == ["10BT", "6VXX"]
    assert fetch.resolution_of({"rcsb_entry_info": {"resolution_combined": [2.1, 2.4]}}) == 2.4
    assert fetch.resolution_of({"rcsb_entry_info": {"resolution_combined": []}}) is None
    assert fetch.resolution_of({}) is None


def test_the_core_table_rows_follow_their_header_before_the_resolution_sources(tmp_path):
    """The resolution-source table must not split the core subset table."""
    def row(subset, pdb):
        return dict(subset=subset, id=f"{subset}/{pdb}", pdb_id=pdb, path=str(tmp_path / pdb), member="",
                    valid=True, missing_residues=0, residues=100, interface_status="pass",
                    max_contact_residues=30, vhh_status="pass", cdr3_lengths=[16], cdr3_sequences=["A" * 16],
                    structure_quality_status="pass", structure_quality_reasons=[], weak_pairs=0, error="",
                    models_first_only=False, legacy_pdb_tail=False, pairs=[])
    rows = [row(s, p) for s, p in (("train_rcsb", "1AAA"), ("sabdab_vhh", "2BBB"),
                                   ("snac_db", "3CCC"), ("test_db55", "4DDD"))]
    audit.RESOLUTION_SOURCE_FILES[:] = [dict(path="entry_resolution.tsv", kind="rcsb_entry_resolution",
                                             sha256="0" * 64)]
    try:
        out = tmp_path / "data_audit_report.md"
        audit.report(tmp_path, rows, [], [], [], 1.0, out)
        lines = out.read_text(encoding="utf-8").splitlines()
    finally:
        audit.RESOLUTION_SOURCE_FILES.clear()
    header = next(i for i, l in enumerate(lines) if l.startswith("| 子集 | 结构文件 |"))
    sources = lines.index("## 分辨率元数据来源")
    # Separator, then the four subset rows, all before the resolution section.
    assert lines[header + 1].startswith("|---")
    assert [lines[header + 2 + k].split("|")[1].strip() for k in range(4)] == \
        ["train_rcsb", "sabdab_vhh", "snac_db", "test_db55"]
    assert sources > header + 5
    assert "`entry_resolution.tsv`" in "\n".join(lines[sources:])
