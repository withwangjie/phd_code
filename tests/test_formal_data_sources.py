"""Regression tests for formal data-source roles and ingestion."""
from __future__ import annotations

import json

import pytest

import nanoqc.data.audit_all_datasets as audit
import nanoqc.data.build_final_pyg_dataset as builder
import nanoqc.data.prepare_external_vhh as prep


def _row(subset: str, pdb: str, suffix: str):
    return dict(subset=subset, pdb_id=pdb, id=f"{subset}/{suffix}")


def test_cross_source_pdb_precedence_is_snac_then_sabdab_then_rcsb():
    rows = [
        _row("train_rcsb", "1ABC", "r1"),
        _row("sabdab_vhh", "1ABC", "s1"),
        _row("snac_db", "1ABC", "n1"),
        _row("snac_db", "1ABC", "n2"),
        _row("train_rcsb", "2DEF", "r2"),
        _row("sabdab_vhh", "2DEF", "s2"),
        _row("train_rcsb", "3GHI", "r3"),
    ]
    kept, dropped = builder.deduplicate_cross_source_pdb(rows)
    kept_ids = {r["id"] for r in kept}
    assert {"snac_db/n1", "snac_db/n2", "sabdab_vhh/s2", "train_rcsb/r3"} <= kept_ids
    assert {(r["id"], preferred) for r, preferred in dropped} == {
        ("train_rcsb/r1", "snac_db"),
        ("sabdab_vhh/s1", "snac_db"),
        ("train_rcsb/r2", "sabdab_vhh"),
    }


def test_sabdab_formal_ingestion_uses_its_own_chain_metadata(tmp_path, monkeypatch):
    summary = tmp_path / "sabdab_nano_summary.tsv"
    summary.write_text(
        "pdb\tresolution\tHchain\tLchain\tantigen_chain\tantigen_type\tscfv\n"
        "1abc\t2.1\tH\tNA\tA\tprotein\tFalse\n",
        encoding="utf-8",
    )
    audit.load_annotations(tmp_path)
    assert audit.sabdab_chain_metadata("1abc") == (["H"], ["A"])
    monkeypatch.setattr(
        audit, "_annotate_sabdab_cdrs",
        lambda sequence: dict(cdr1="QQQ", cdr2="QQ", cdr3="QQQQ", method="test_imgt"),
    )
    monkeypatch.setattr(audit, "_ig_variable_domain", lambda sequence: (False, "test"))
    chains = [dict(name="H", sequence="Q" * 90), dict(name="A", sequence="M" * 120)]
    features, allowed = audit.sabdab_features(chains, "1ABC")
    assert features["vhh_status"] == "pass"
    assert features["vhh_chain"] == "H"
    assert features["cdr3_sequences"] == ["QQQQ"]
    assert features["cdr_annotation_method"] == "test_imgt"
    assert features["sabdab_antigen_chains"] == ["A"]
    assert allowed("H", "A") and allowed("A", "H")
    assert not allowed("H", "B")


def test_sabdab_rejects_light_chain_and_unannotated_ig(monkeypatch):
    audit.SABDAB_ANNOTATIONS.clear()
    audit.SABDAB_ANNOTATIONS["1ABC"] = [
        dict(Hchain="H", Lchain="L", antigen_chain="A", antigen_type="protein", scfv="False")
    ]
    failed, _ = audit.sabdab_features(
        [dict(name="H", sequence="Q" * 90), dict(name="A", sequence="M" * 120)], "1ABC")
    assert failed["vhh_status"] == "fail"
    assert "轻链" in failed["vhh_reason"]

    audit.SABDAB_ANNOTATIONS["1ABC"] = [
        dict(Hchain="H", Lchain="NA", antigen_chain="A", antigen_type="protein", scfv="False")
    ]
    monkeypatch.setattr(
        audit, "_annotate_sabdab_cdrs",
        lambda sequence: dict(cdr1="", cdr2="", cdr3="QQQQ", method="test"),
    )
    monkeypatch.setattr(
        audit, "_ig_variable_domain",
        lambda sequence: (sequence.startswith("I"), "test_detector"),
    )
    failed, _ = audit.sabdab_features([
        dict(name="H", sequence="Q" * 90),
        dict(name="A", sequence="M" * 120),
        dict(name="B", sequence="I" * 90),
    ], "1ABC")
    assert failed["vhh_status"] == "fail"
    assert failed["other_antibody_chains"] == ["B"]


def test_generic_rcsb_is_audit_only_not_formal_training():
    assert builder.FORMAL_TRAIN_SOURCES == ("snac_db", "sabdab_vhh")
    assert builder.SOURCE_ROLE["train_rcsb"] == "audit_only"
    assert "train_rcsb" not in audit.FORMAL_VHH_SUBSETS


def test_study_foldseek_universe_contains_only_verified_formal_vhh(tmp_path):
    ledger = tmp_path / "data_audit_details.jsonl"
    pass_fields = dict(valid=True, missing_residues=0, interface_status="pass",
                       max_contact_residues=20, vhh_status="pass",
                       structure_quality_status="pass")
    rows = [
        dict(subset="snac_db", pdb_id="1ABC", **pass_fields),
        dict(subset="sabdab_vhh", pdb_id="2DEF", **pass_fields),
        dict(subset="train_rcsb", pdb_id="3GHI", **pass_fields),
        dict(subset="test_db55", pdb_id="4JKL", **pass_fields),
        dict(subset="sabdab_vhh", pdb_id="5MNO", **{**pass_fields, "vhh_status": "fail"}),
        dict(subset="extra_misc", pdb_id="6PQR", **pass_fields),
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert prep.study_pdb_ids(ledger) == ["1abc", "2def"]


def test_graph_protocol_version_changes_with_data_admission_semantics():
    assert builder.VERSION == "1.11"


def test_cluster_universe_cannot_be_bridged_by_qc_failures_or_lower_priority_sources():
    good = dict(valid=True, missing_residues=0, interface_status="pass",
                max_contact_residues=20, vhh_status="pass",
                structure_quality_status="pass")
    rows = [
        # SNAC wins by source precedence but fails QC; the passing SAbDab copy
        # must not rescue the PDB into the clustering universe.
        dict(subset="snac_db", pdb_id="1AAA", **{**good, "structure_quality_status": "fail"}),
        dict(subset="sabdab_vhh", pdb_id="1AAA", **good),
        dict(subset="sabdab_vhh", pdb_id="2BBB", **good),
        dict(subset="snac_db", pdb_id="3CCC", **{**good, "missing_residues": 1}),
        dict(subset="snac_db", pdb_id="4DDD", **{**good, "max_contact_residues": 10}),
        dict(subset="snac_db", pdb_id="5EEE", **{**good, "interface_status": "weak"}),
    ]
    assert audit.formal_source_by_pdb(rows)["1AAA"] == "snac_db"
    assert audit.formal_clustering_pdb_ids(rows, 15) == ["2bbb"]


def test_foldseek_formal_sources_drop_qc_failures(tmp_path):
    ledger = tmp_path / "data_audit_details.jsonl"
    base = dict(path="/tmp/x.pdb", member="", valid=True, missing_residues=0,
                interface_status="pass", max_contact_residues=20, vhh_status="pass",
                structure_quality_status="pass")
    rows = [
        dict(base, id="good", subset="sabdab_vhh", pdb_id="2BBB"),
        dict(base, id="bad", subset="snac_db", pdb_id="3CCC",
             structure_quality_status="fail"),
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    from nanoqc.data.build_foldseek_pairs import audit_sources
    sources = audit_sources(ledger, formal_only=True, min_interface_residues=15)
    assert [row["id"] for row in sources["2bbb"]] == ["good"]
    assert sources["3ccc"] == []


def test_raw_structure_change_after_audit_fails_closed(tmp_path):
    raw = tmp_path / "1abc.pdb"
    raw.write_text("ORIGINAL\n", encoding="utf-8")
    task = dict(path=str(raw), member="", subset="snac_db", id="snac/1abc")
    expected = audit.task_source_sha256(dict(task))
    row = dict(task, pdb_id="1ABC", source_structure_sha256=expected)
    assert builder.verify_audited_source(row) == expected

    raw.write_text("CHANGED\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw structure changed since data audit"):
        builder.verify_audited_source(row)


def test_zip_member_hash_binds_member_not_container_path(tmp_path):
    import zipfile
    archive = tmp_path / "nb_complexes.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("1abc.pdb", "MEMBER-A\n")
        handle.writestr("2def.pdb", "MEMBER-B\n")
    first = audit.task_source_sha256(
        dict(path=str(archive), member="1abc.pdb", subset="snac_db", id="a"))
    second = audit.task_source_sha256(
        dict(path=str(archive), member="2def.pdb", subset="snac_db", id="b"))
    assert first != second
    assert len(first) == len(second) == 64


def test_foldseek_rejects_raw_structure_changed_after_audit(tmp_path):
    from nanoqc.data.build_foldseek_pairs import antigen_structure
    raw = tmp_path / "2bbb.pdb"
    raw.write_text("ORIGINAL\n", encoding="utf-8")
    task = dict(path=str(raw), member="", subset="sabdab_vhh", id="sabdab/2bbb")
    expected = audit.task_source_sha256(dict(task))
    source = dict(task, source_structure_sha256=expected)
    raw.write_text("CHANGED\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw structure changed since data audit"):
        antigen_structure("2bbb", [source], [], 20)
