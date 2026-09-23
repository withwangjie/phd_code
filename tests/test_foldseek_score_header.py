"""A numeric fident column must never be accepted as a TM-score."""

import json
import sys
from unittest.mock import patch

from nanoqc.data.build_independence_cluster_map import main


def test_foldseek_requires_length_normalized_tm_score(tmp_path):
    pairs = tmp_path / "pairs.tsv"
    output = tmp_path / "clusters.json"
    argv = ["cluster-map", "--pairs", str(pairs), "--out-json", str(output),
            "--min-score", "0.5", "--score-semantics", "qtmscore"]

    pairs.write_text("query\ttarget\tfident\n1abc\t2def\t0.9\n", encoding="utf-8")
    with patch.object(sys, "argv", argv):
        try:
            main()
        except ValueError as exc:
            assert "require qtmscore or ttmscore" in str(exc)
        else:
            raise AssertionError("Foldseek fident was accepted as TM-score")

    pairs.write_text("query\ttarget\tqtmscore\n1abc\t2def\t0.6\n", encoding="utf-8")
    with patch.object(sys, "argv", argv):
        assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["1abc"] == json.loads(output.read_text(encoding="utf-8"))["2def"]
    provenance = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert provenance["score_field"] == "qtmscore"
    assert provenance["score_header_validated"] is True


def test_foldseek_rejects_malformed_similarity_row(tmp_path):
    pairs=tmp_path/"pairs.tsv"
    output=tmp_path/"clusters.json"
    pairs.write_text("query\ttarget\tqtmscore\n1abc\t2def\tbad\n",encoding="utf-8")
    argv=["cluster-map","--pairs",str(pairs),"--out-json",str(output),
          "--min-score","0.5","--score-semantics","qtmscore"]
    with patch.object(sys,"argv",argv):
        try:
            main()
        except ValueError as exc:
            assert "nonnumeric qtmscore" in str(exc)
        else:
            raise AssertionError("Malformed similarity row was silently skipped")
    assert not output.exists()


def test_foldseek_rejects_invalid_pdb_identifier(tmp_path):
    pairs=tmp_path/"pairs.tsv"
    pairs.write_text("query\ttarget\tqtmscore\nx\t2def\t0.8\n",encoding="utf-8")
    argv=["cluster-map","--pairs",str(pairs),"--out-json",str(tmp_path/"map.json"),
          "--min-score","0.5","--score-semantics","qtmscore"]
    with patch.object(sys,"argv",argv):
        try:
            main()
        except ValueError as exc:
            assert "invalid four-character PDB identifier" in str(exc)
        else:
            raise AssertionError("Invalid PDB identifier was accepted")


def test_foldseek_does_not_truncate_pdb_identifier(tmp_path):
    pairs=tmp_path/"pairs.tsv"
    pairs.write_text("query\ttarget\tqtmscore\n1abc_extra\t2def\t0.8\n",encoding="utf-8")
    argv=["cluster-map","--pairs",str(pairs),"--out-json",str(tmp_path/"map.json"),
          "--min-score","0.5","--score-semantics","qtmscore"]
    with patch.object(sys,"argv",argv):
        try:
            main()
        except ValueError as exc:
            assert "Invalid PDB identifier" in str(exc)
        else:
            raise AssertionError("Malformed PDB identifier was silently truncated")
