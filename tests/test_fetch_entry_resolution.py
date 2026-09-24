"""fetch_entry_resolution against RCSB's real response shape (PROTOCOL_AMENDMENTS.md A15).

The fixture is the verbatim body RCSB returned for
``{entries(entry_ids:["10BT","10CY","6VXX"]){rcsb_id rcsb_entry_info{resolution_combined} exptl{method}}}``.
"""
from __future__ import annotations

import io
import os
import json
import urllib.error

import pytest

from nanoqc.data import fetch_entry_resolution as fetch

REAL_RESPONSE = (
    '{"data":{"entries":[{"rcsb_id":"10BT","rcsb_entry_info":{"resolution_combined":[1.99]},'
    '"exptl":[{"method":"X-RAY DIFFRACTION"}]},{"rcsb_id":"10CY","rcsb_entry_info":'
    '{"resolution_combined":[2.49]},"exptl":[{"method":"X-RAY DIFFRACTION"}]},'
    '{"rcsb_id":"6VXX","rcsb_entry_info":{"resolution_combined":[2.8]},'
    '"exptl":[{"method":"ELECTRON MICROSCOPY"}]}]}}'
)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _server(bodies, calls=None):
    """urlopen stub returning each body in turn; records the decoded request payloads."""
    remaining = list(bodies)

    def urlopen(request, timeout=None):
        if calls is not None:
            calls.append(json.loads(request.data.decode("utf-8"))["query"])
        body = remaining.pop(0)
        if isinstance(body, Exception):
            raise body
        return _Response(body.encode("utf-8"))

    return urlopen


def test_ids_come_from_lines_and_from_assembly_file_names():
    assert fetch.entry_ids(["10bt", "  6VXX ", "10bt_assembly1.cif", "", "junk"]) == ["10BT", "6VXX"]
    assert fetch.entry_ids(["pdb_000010zo.cif", "pdb_00001ol0.cif", "pdb1abc.ent"]) == [
        "10ZO", "1OL0", "1ABC"]


def test_resolution_is_the_worst_reported_value_or_none():
    assert fetch.resolution_of({"rcsb_entry_info": {"resolution_combined": [1.99]}}) == 1.99
    assert fetch.resolution_of({"rcsb_entry_info": {"resolution_combined": [2.1, 2.4]}}) == 2.4
    for empty in ({"rcsb_entry_info": {"resolution_combined": []}},
                  {"rcsb_entry_info": {"resolution_combined": None}},
                  {"rcsb_entry_info": {}}, {}):
        assert fetch.resolution_of(empty) is None
    assert fetch.methods_of({"exptl": [{"method": "X-RAY DIFFRACTION"}]}) == "X-RAY DIFFRACTION"
    assert fetch.methods_of({}) == ""


def test_end_to_end_writes_the_table_the_audit_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.urllib.request, "urlopen", _server([REAL_RESPONSE]))
    ids = tmp_path / "ids.txt"
    ids.write_text("10bt\n10cy\n6vxx\n")
    out = tmp_path / "rcsb_entry_resolution.tsv"
    assert fetch.main(["--ids", str(ids), "--out", str(out)]) == 0
    assert out.read_text() == ("pdb\tresolution\tmethod\n"
                               "10BT\t1.99\tX-RAY DIFFRACTION\n"
                               "10CY\t2.49\tX-RAY DIFFRACTION\n"
                               "6VXX\t2.8\tELECTRON MICROSCOPY\n")
    assert not list(tmp_path.glob("*.part"))

    # The audit reads exactly this file, by PDB ID.
    from nanoqc.data import audit_all_datasets as audit
    audit.load_annotations(tmp_path)
    try:
        assert audit.RESOLUTION_BY_PDB["10BT"] == (1.99, "rcsb_entry_resolution")
        assert audit.RESOLUTION_BY_PDB["6VXX"] == (2.8, "rcsb_entry_resolution")
        assert [f["kind"] for f in audit.RESOLUTION_SOURCE_FILES] == ["rcsb_entry_resolution"]
    finally:
        audit.load_annotations(tmp_path / "none")


def test_an_entry_without_a_resolution_is_written_empty_and_stays_excluded(tmp_path, monkeypatch):
    body = ('{"data":{"entries":[{"rcsb_id":"1NMR","rcsb_entry_info":{"resolution_combined":null},'
            '"exptl":[{"method":"SOLUTION NMR"}]}]}}')
    monkeypatch.setattr(fetch.urllib.request, "urlopen", _server([body]))
    out = tmp_path / "rcsb_entry_resolution.tsv"
    ids = tmp_path / "ids.txt"
    ids.write_text("1nmr\n")
    assert fetch.main(["--ids", str(ids), "--out", str(out)]) == 0
    assert out.read_text().splitlines()[1] == "1NMR\t\tSOLUTION NMR"
    from nanoqc.data import audit_all_datasets as audit
    audit.load_annotations(tmp_path)
    try:
        assert "1NMR" not in audit.RESOLUTION_BY_PDB
    finally:
        audit.load_annotations(tmp_path / "none")


def test_requests_are_chunked_and_cover_every_id(tmp_path, monkeypatch):
    queries = []
    bodies = ['{"data":{"entries":[{"rcsb_id":"10BT","rcsb_entry_info":{"resolution_combined":[1.99]}}]}}',
              '{"data":{"entries":[{"rcsb_id":"6VXX","rcsb_entry_info":{"resolution_combined":[2.8]}}]}}']
    monkeypatch.setattr(fetch.urllib.request, "urlopen", _server(bodies, queries))
    ids = tmp_path / "ids.txt"
    ids.write_text("10bt\n6vxx\n")
    out = tmp_path / "rcsb_entry_resolution.tsv"
    assert fetch.main(["--ids", str(ids), "--out", str(out), "--chunk", "1"]) == 0
    assert len(queries) == 2 and '"10BT"' in queries[0] and '"6VXX"' in queries[1]
    assert len(out.read_text().splitlines()) == 3


def test_a_transient_failure_is_retried_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fetch.urllib.request, "urlopen",
                        _server([urllib.error.URLError("reset"), REAL_RESPONSE]))
    ids = tmp_path / "ids.txt"
    ids.write_text("10bt\n10cy\n6vxx\n")
    out = tmp_path / "rcsb_entry_resolution.tsv"
    assert fetch.main(["--ids", str(ids), "--out", str(out)]) == 0
    assert "10BT\t1.99" in out.read_text()


def test_an_interrupted_fetch_keeps_what_it_got_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    first = '{"data":{"entries":[{"rcsb_id":"10BT","rcsb_entry_info":{"resolution_combined":[1.99]}}]}}'
    monkeypatch.setattr(fetch.urllib.request, "urlopen",
                        _server([first] + [urllib.error.URLError("down")] * 4))
    ids = tmp_path / "ids.txt"
    ids.write_text("10bt\n6vxx\n")
    out = tmp_path / "rcsb_entry_resolution.tsv"
    with pytest.raises(SystemExit):
        fetch.main(["--ids", str(ids), "--out", str(out), "--chunk", "1"])
    assert "10BT\t1.99" in out.read_text()  # the finished chunk survived

    second = '{"data":{"entries":[{"rcsb_id":"6VXX","rcsb_entry_info":{"resolution_combined":[2.8]}}]}}'
    queries = []
    monkeypatch.setattr(fetch.urllib.request, "urlopen", _server([second], queries))
    assert fetch.main(["--ids", str(ids), "--out", str(out), "--chunk", "1", "--resume"]) == 0
    assert queries == [fetch.QUERY % '"6VXX"']  # 10BT is not fetched again
    assert len(out.read_text().splitlines()) == 3


def test_a_rejected_query_or_a_foreign_payload_fails_loudly(tmp_path, monkeypatch):
    ids = tmp_path / "ids.txt"
    ids.write_text("10bt\n")
    out = tmp_path / "rcsb_entry_resolution.tsv"
    for body, expected in (('{"errors":[{"message":"bad field"}]}', "rejected the query"),
                           ('{"data":{}}', "no data.entries")):
        monkeypatch.setattr(fetch.urllib.request, "urlopen", _server([body]))
        with pytest.raises(SystemExit) as raised:
            fetch.main(["--ids", str(ids), "--out", str(out)])
        assert expected in str(raised.value)


def test_the_output_name_must_be_the_one_the_audit_reads(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("10bt\n")
    with pytest.raises(SystemExit) as raised:
        fetch.main(["--ids", str(ids), "--out", str(tmp_path / "resolutions.tsv")])
    assert raised.value.code == 2


def _audit_dir(tmp_path, rows):
    import json as _json
    d = tmp_path / "audit"
    d.mkdir(exist_ok=True)
    (d / "data_audit_details.jsonl").write_text("".join(_json.dumps(r) + "\n" for r in rows))
    return d


def test_the_audit_says_which_entries_need_a_resolution(tmp_path, monkeypatch):
    """One table for every subset: ask the finished audit, do not guess."""
    audit_dir = _audit_dir(tmp_path, [
        dict(subset="train_rcsb", pdb_id="10BT", valid=True, resolution_angstrom=None),
        dict(subset="sabdab_vhh", pdb_id="6VXX", valid=True, resolution_angstrom=None),
        dict(subset="snac_db", pdb_id="1MEL", valid=True, resolution_angstrom=2.5),   # already has one
        dict(subset="train_rcsb", pdb_id="9BAD", valid=False, resolution_angstrom=None),  # unreadable
    ])
    queries = []
    monkeypatch.setattr(fetch.urllib.request, "urlopen", _server([REAL_RESPONSE], queries))
    out = tmp_path / "entry_resolution.tsv"
    assert fetch.main(["--missing-from-audit", str(audit_dir), "--out", str(out)]) == 0
    assert queries == [fetch.QUERY % '"10BT","6VXX"']


def test_the_fetch_can_be_restricted_to_one_subset(tmp_path, monkeypatch):
    audit_dir = _audit_dir(tmp_path, [
        dict(subset="train_rcsb", pdb_id="10BT", valid=True, resolution_angstrom=None),
        dict(subset="sabdab_vhh", pdb_id="6VXX", valid=True, resolution_angstrom=None),
    ])
    queries = []
    monkeypatch.setattr(fetch.urllib.request, "urlopen", _server([REAL_RESPONSE], queries))
    out = tmp_path / "entry_resolution.tsv"
    assert fetch.main(["--missing-from-audit", str(audit_dir), "--subset", "train_rcsb",
                       "--out", str(out)]) == 0
    assert queries == [fetch.QUERY % '"10BT"']


def test_with_no_arguments_it_finds_the_latest_audit_and_the_data_root(tmp_path, monkeypatch):
    """The whole job for this repository: no paths to type, nothing to guess."""
    monkeypatch.setattr(fetch, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("QP_DATA_ROOT", raising=False)
    (tmp_path / "data").mkdir()
    for name, pdb in (("run_old", "9OLD"), ("run_new", "10BT")):
        d = tmp_path / "runs" / name / "audit"
        d.mkdir(parents=True)
        (d / "data_audit_details.jsonl").write_text(
            json.dumps(dict(subset="train_rcsb", pdb_id=pdb, valid=True, resolution_angstrom=None)) + "\n")
    newest = tmp_path / "runs" / "run_new" / "audit" / "data_audit_details.jsonl"
    newest.touch()
    assert fetch.latest_audit_dir() == newest.parent
    assert fetch.data_root() == tmp_path / "data"

    queries = []
    monkeypatch.setattr(fetch.urllib.request, "urlopen", _server([REAL_RESPONSE], queries))
    assert fetch.main([]) == 0
    assert queries == [fetch.QUERY % '"10BT"']
    assert (tmp_path / "data" / "entry_resolution.tsv").is_file()


def test_the_data_root_follows_the_configured_one(tmp_path, monkeypatch):
    monkeypatch.setenv("QP_DATA_ROOT", str(tmp_path / "elsewhere"))
    assert fetch.data_root() == tmp_path / "elsewhere"


def test_it_runs_as_a_plain_file_from_any_directory(tmp_path):
    """python .../fetch_entry_resolution.py works without PYTHONPATH or an install."""
    import subprocess
    import sys as _sys
    from pathlib import Path as _P

    script = _P(fetch.__file__).resolve()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([_sys.executable, str(script), "--help"], cwd=str(script.parent),
                          capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert "entry_resolution.tsv" in done.stdout
