"""Regression tests for formal data-source roles and ingestion."""
from __future__ import annotations

import json

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
    rows = [
        dict(subset="snac_db", pdb_id="1ABC", valid=True, vhh_status="pass"),
        dict(subset="sabdab_vhh", pdb_id="2DEF", valid=True, vhh_status="pass"),
        dict(subset="train_rcsb", pdb_id="3GHI", valid=True, vhh_status="not_applicable"),
        dict(subset="test_db55", pdb_id="4JKL", valid=True, vhh_status="not_applicable"),
        dict(subset="sabdab_vhh", pdb_id="5MNO", valid=True, vhh_status="fail"),
        dict(subset="extra_misc", pdb_id="6PQR", valid=True, vhh_status="not_applicable"),
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert prep.study_pdb_ids(ledger) == ["1abc", "2def"]


def test_graph_protocol_version_changes_with_data_admission_semantics():
    assert builder.VERSION == "1.9"
