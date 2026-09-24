"""SAbDab2 single-domain CSV -> classic SAbDab summary TSV."""
from __future__ import annotations

import json

import pytest

from nanoqc.data import convert_sabdab2_summary as conv
from nanoqc.data import select_external_vhh_candidates as sel

HEADER = ("INSTANCE,PDB,Hchain,Lchain,VH,CDR-H1,CDR-H2,CDR-H3,model,antigen_chain,antigen_type,"
          "antigen_name,date,method,resolution,type")
ROWS = [
    "pdb_000010zo-A,pdb_000010zo,A,,EVKLQ,PEVF,ITPW,AQGWGIASMRY,0,B,PROTEIN,Nucleoprotein,2026/02/12,XRAY,1.65,SD-H",
    "pdb_000010zo-C,pdb_000010zo,C,,EVKLQ,PEVF,ITPW,AQGWGIASMRY,0,D,PROTEIN,Nucleoprotein,2026/02/12,XRAY,1.65,SD-H",
    "pdb_00007xyz-N,pdb_00007xyz,N,,QVQLV,GRTF,ISWS,AAGRY,0,A | B,PROTEIN|SUGAR,Spike,2025/11/03,"
    "ELECTRON_MICROSCOPY,3.40,SD-H",
    "pdb_00009abc-L,pdb_00009abc,L,,DIQMT,RASQ,SASF,QQHYT,0,A,PROTEIN,Thing,2025/01/01,XRAY,2.00,SD-L",
    "pdb_00009def-S,pdb_00009def,S,,ARVDQ,GFSL,IYSG,TIGGSLSV,0,A,PROTEIN,Thing,2025/01/02,XRAY,2.00,VNAR",
]


def _write(tmp_path, rows=ROWS, header=HEADER):
    path = tmp_path / "sabdab2.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_identifiers_dates_and_vocabularies_are_mapped():
    assert conv.pdb_id("pdb_000010zo") == "10zo" and conv.pdb_id(" 8G2Y ") == "8g2y"
    assert conv.iso_date("2026/02/12") == "2026-02-12"
    row = conv.convert_row(dict(PDB="pdb_00007xyz", Hchain="N", Lchain="", antigen_chain="A | B",
                                antigen_type="PROTEIN|SUGAR", antigen_name="Spike", date="2025/11/03",
                                method="ELECTRON_MICROSCOPY", resolution="", VH="Q"))
    assert row["pdb"] == "7xyz" and row["method"] == "ELECTRON MICROSCOPY" and row["scfv"] == "False"
    # Empty fields use the classic file's NA; antigen tokens are lower-cased, never dropped.
    assert row["Lchain"] == "NA" and row["resolution"] == "NA" and row["antigen_type"] == "protein | sugar"
    for bad in ("pdb_0000", "xx", "pdb_000010z!"):
        with pytest.raises(ValueError, match="four-character"):
            conv.pdb_id(bad)


def test_only_the_requested_single_domain_class_is_kept(tmp_path):
    rows, stats = conv.convert(_write(tmp_path), ["SD-H"])
    assert stats["rows_by_type"] == {"SD-H": 3, "SD-L": 1, "VNAR": 1}
    assert {row["pdb"] for row in rows} == {"10zo", "7xyz"}
    assert stats["failures"] == []
    both, _ = conv.convert(_write(tmp_path), ["SD-H", "VNAR"])
    assert {row["pdb"] for row in both} == {"10zo", "7xyz", "9def"}


def test_a_file_that_is_not_a_sabdab2_export_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a SAbDab2 single-domain export"):
        conv.convert(_write(tmp_path, rows=["8g2y,N,NA"], header="pdb,Hchain,Lchain"), ["SD-H"])


def test_converted_file_feeds_the_selector_unchanged(tmp_path):
    import datetime as dt
    out = tmp_path / "converted.tsv"
    assert conv.main(["--source", str(_write(tmp_path)), "--out", str(out)]) == 0
    manifest = json.loads(out.with_suffix(".conversion.json").read_text())
    assert manifest["entries"] == 2 and manifest["date_range"] == ["2025-11-03", "2026-02-12"]
    assert "Fab" in manifest["caveat"]

    grouped = sel.read_sabdab([out])
    assert set(grouped) == {"10zo", "7xyz"}
    cutoff = dt.date(2025, 1, 1)
    entry = sel.metadata_gate("10zo", grouped["10zo"], released_after=cutoff, excluded=set(), max_resolution=3.0)
    # Both VHH copies of the entry are offered to the structure-level choice.
    assert entry["reasons"] == [] and entry["vhh_chains"] == ["A", "C"]
    assert entry["release_date"] == "2026-02-12" and entry["resolution"] == 1.65
    mixed = sel.metadata_gate("7xyz", grouped["7xyz"], released_after=cutoff, excluded=set(), max_resolution=3.0)
    # "protein | sugar" still has a polypeptide antigen chain; only resolution rejects it.
    assert mixed["reasons"] == ["resolution_above_limit"]


def test_an_unconverted_or_empty_summary_fails_loudly(tmp_path):
    """A silently empty table is indistinguishable from 'every entry was rejected'."""
    raw = _write(tmp_path)  # the SAbDab2 CSV itself, not the converted TSV
    with pytest.raises(ValueError, match="convert it first with nanoqc.data.convert_sabdab2_summary"):
        sel.read_sabdab([raw])
    other = tmp_path / "other.tsv"
    other.write_text("name\tvalue\nx\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'pdb' column"):
        sel.read_sabdab([other])
    empty = tmp_path / "empty.tsv"
    empty.write_text("pdb\tHchain\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no row carries a four-character PDB ID"):
        sel.read_sabdab([empty])
