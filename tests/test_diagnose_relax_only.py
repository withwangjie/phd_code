"""The relax-only diagnostic selects, parallelises, records and resumes seeds."""
import json
from pathlib import Path

import yaml

from nanoqc.experiments import relax_only_diagnostic as diag


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run).mkdir()
    (run / "frozen_config.yaml").write_text(yaml.safe_dump(
        {"hardware": {"openmm_platform": "Reference", "openmm_device": "0"}}))
    for target, seed, status in (("7nxx", 43, "failed"), ("7zra", 44, "failed"), ("4aq1", 42, "passed")):
        exp = run / "validation_queue" / "results" / target / f"seed_{seed}" / "experiment"
        exp.mkdir(parents=True)
        (exp.parent / "experiment.json").write_text(json.dumps(
            {"input_structure": "missing.cif", "active_residues": ["H:55"]}))
        (exp / "run_manifest.json").write_text(json.dumps(
            {"arguments": {"seed": seed, "relax_iterations": 1000}}))
        (exp / "relax_only_result.json").write_text(json.dumps(
            {"evaluation_status": status, "evaluation_failure_reasons": [],
             "movable_force_rms_kj_mol_nm": 1.0, "relaxation": {}}))
    return run


def test_failed_seeds_run_in_worker_processes_errors_are_recorded_and_resume(tmp_path, capsys):
    run = _fake_run(tmp_path)
    out = tmp_path / "out"
    assert diag.main([str(run), "--out", str(out), "--workers-per-gpu", "2"]) == 1
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert sorted(r["seed"] for r in rows) == ["7nxx/seed_43", "7zra/seed_44"]
    assert all(r["outcome"] == "error" and "missing.cif" in r["new"]["error"] for r in rows)
    assert "2 worker processes" in capsys.readouterr().out
    # A second invocation resumes: nothing is rerun, the summary is rebuilt.
    assert diag.main([str(run), "--out", str(out)]) == 1
    assert len((out / "results.jsonl").read_text().splitlines()) == 2
    assert "0 to run" in capsys.readouterr().out
    assert json.loads((out / "summary.json").read_text())["counts"] == {"error": 2}


def test_explicit_seeds_and_out_inside_run_are_checked(tmp_path):
    run = _fake_run(tmp_path)
    import pytest
    with pytest.raises(SystemExit):
        diag.main([str(run), "--out", str(run / "diag")])
    with pytest.raises(SystemExit):
        diag.main([str(run), "--out", str(tmp_path / "o"), "--seeds", "9xyz/seed_1"])
    assert [diag._key(d) for d in diag._seed_dirs(run)] == ["4aq1/seed_42", "7nxx/seed_43", "7zra/seed_44"]
