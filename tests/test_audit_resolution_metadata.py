"""Entry resolution from curation metadata; pipeline staging is never audited."""
from __future__ import annotations

from nanoqc.data import audit_all_datasets as audit

SNAC_HEADER = "Name,Resolution,PDB_ID,Chain_VH,Chain_VH_old_id,Chain_Ag\n"


def test_parse_resolution_reads_the_first_number_or_none():
    assert audit.parse_resolution("2.5") == 2.5
    assert audit.parse_resolution("2.5, 2.7") == 2.5
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
