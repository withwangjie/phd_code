"""A formal run builds and freezes its own structure-similarity search."""

from __future__ import annotations

import json
from pathlib import Path

from nanoqc.common.repo_io import sha256_file
from nanoqc.pipeline import run_full_experiment as full


def test_foldseek_is_built_once_per_run_and_bound_to_its_audit(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    config = {
        "paths": {"repo_root": str(tmp_path), "data_root": str(data)},
        "data_audit": {"workers": 4},
        "runtime_resolution": {"foldseek_executable": "/opt/foldseek/bin/foldseek"},
    }
    resolved = []
    def which(name):
        resolved.append(name)
        return "/usr/bin/foldseek"
    monkeypatch.setattr(full.shutil, "which", which)
    calls = []

    def make_run(name: str, universe_text: str):
        run = tmp_path / name
        audit = run / "audit"
        audit.mkdir(parents=True)
        universe = audit / "cluster_universe.txt"
        universe.write_text(universe_text)
        ledger = audit / "data_audit_details.jsonl"
        ledger.write_text('{"subset":"sabdab_vhh"}\n')
        orchestrator = full.Orchestrator(config, run)

        def build(stage, argv):
            assert stage == "queue_freeze_foldseek"
            calls.append((run, list(argv)))
            assert argv[argv.index("--universe") + 1] == str(universe)
            assert argv[argv.index("--audit-dir") + 1] == str(audit)
            assert argv[argv.index("--foldseek") + 1] == "/usr/bin/foldseek"
            output = Path(argv[argv.index("--out") + 1])
            output.write_text("query\ttarget\tmintmscore\n1abc\t1abc\t1.0000\n")
            manifest = {
                "schema": "foldseek_pairs_v1",
                "score": "mintmscore",
                "pair_table_sha256": sha256_file(output),
                "universe_sha256": sha256_file(universe),
                "audit_ledger_sha256": sha256_file(ledger),
                "min_interface_residues": 15,
            }
            output.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            return 0, run / "logs" / "queue_freeze_foldseek.log"

        monkeypatch.setattr(orchestrator, "_run_subprocess", build)
        return orchestrator, universe, ledger

    first, universe, ledger = make_run("run_one", "1abc\n")
    assert first._ensure_run_local_foldseek_pairs(universe, ledger, 15)[0]
    assert resolved == ["/opt/foldseek/bin/foldseek"]
    assert first._ensure_run_local_foldseek_pairs(universe, ledger, 15)[0]
    assert len(calls) == 1

    first.run_local_foldseek_pairs().write_text("tampered\n")
    ok, reason = first._ensure_run_local_foldseek_pairs(universe, ledger, 15)
    assert not ok and "pair_table_sha256" in reason

    second, second_universe, second_ledger = make_run("run_two", "1abc\n2def\n")
    assert second._ensure_run_local_foldseek_pairs(second_universe, second_ledger, 15)[0]
    assert len(calls) == 2
    assert first.run_local_foldseek_pairs() != second.run_local_foldseek_pairs()

    # A crash between writing the TSV and manifest can be retried safely.
    second.run_local_foldseek_pairs().with_suffix(".manifest.json").unlink()
    assert second._ensure_run_local_foldseek_pairs(second_universe, second_ledger, 15)[0]
    assert len(calls) == 3
    assert "--force" in calls[-1][1]
