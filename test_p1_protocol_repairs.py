"""Regression tests for P1 protocol repairs."""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import batch_benchmark_hard_set as bbh
import run_real_complex_pilot as rcp
from run_full_experiment import Orchestrator, load_config


def _metric(solver: str, *, termination_reason=None, gap=1.0, hit=0,
            objective=None, restarts=None) -> dict:
    return dict(
        solver=solver,
        outputs=1000,
        gap=gap,
        hit=hit,
        ground_probability=0.0,
        low_energy_mass=0.1,
        low_energy_coverage=0.2,
        entropy=1.0,
        solver_seconds=1.0,
        qaoa_objective=objective,
        qaoa_restarts=restarts,
        termination_reason=termination_reason,
    )


def _write_case(root: Path, qaoa_reason: str) -> None:
    cases = root / "cases"
    cases.mkdir(parents=True)
    metrics = [
        _metric("qaoa", termination_reason=qaoa_reason, gap=0.5, hit=1,
                objective="cvar", restarts=4),
        _metric("sa", gap=1.0),
        _metric("uniform", gap=1.5),
        _metric("greedy", gap=0.8),
    ]
    (cases / "case.json").write_text(
        json.dumps({"config": {"pdb_id": "1abc"}, "metrics": metrics}),
        encoding="utf-8",
    )


def test_single_target_run_has_no_hidden_4s10_fallback() -> None:
    source = inspect.getsource(rcp.main)
    assert "requested_pdb=args.pdb_id" in source
    assert "targets==1" not in source
    assert "'4s10' if args.eval_shots" not in source


def test_all_restarts_failed_is_excluded_from_inference(tmp_path: Path) -> None:
    root = tmp_path / "failed"
    _write_case(root, "all_restarts_failed")
    status = bbh._paired_statistics_main([
        "--results-dir", str(root),
        "--budget-mode", "outputs",
        "--primary-outputs", "1000",
        "--primary-objective", "cvar",
        "--primary-restarts", "4",
        "--resamples", "100",
    ])
    assert status == 0
    payload = json.loads((root / "statistics_outputs.json").read_text(encoding="utf-8"))
    assert payload["exclusions"]["qaoa:all_restarts_failed"] == 1
    assert payload["effects"] == []


def test_fixed_budget_max_evaluations_reached_remains_analyzable(tmp_path: Path) -> None:
    root = tmp_path / "budget"
    _write_case(root, "max_evaluations_reached")
    status = bbh._paired_statistics_main([
        "--results-dir", str(root),
        "--budget-mode", "outputs",
        "--primary-outputs", "1000",
        "--primary-objective", "cvar",
        "--primary-restarts", "4",
        "--resamples", "100",
    ])
    assert status == 0
    payload = json.loads((root / "statistics_outputs.json").read_text(encoding="utf-8"))
    assert payload["exclusions"].get("qaoa:all_restarts_failed", 0) == 0
    assert payload["paired_cases"]["sa"] == 1
    assert payload["effects"]


def test_formal_config_freezes_primary_contrast_and_explicit_smoke_target() -> None:
    cfg = load_config(Path("full_experiment_config.yaml"))
    assert cfg["smoke_check"]["recovery_pilot_pdb_id"].lower() == "4s10"
    assert cfg["statistics"]["primary_outputs"] == 1000
    assert cfg["statistics"]["primary_objective"] == "cvar"
    assert cfg["statistics"]["primary_restarts"] == 4
    assert cfg["statistics"]["primary_objective"] in cfg["qc_benchmark"]["qaoa_objective"]
    assert cfg["statistics"]["primary_restarts"] in cfg["qc_benchmark"]["qaoa_restarts"]
    assert cfg["statistics"]["primary_outputs"] in cfg["qc_benchmark"]["outputs"]


def test_stage_timeout_returns_124(tmp_path: Path) -> None:
    class Dummy:
        config = {
            "hardware": {},
            "control": {"stage_timeout_seconds": 0.1},
        }
        repo_root = tmp_path
        log_dir = tmp_path

    started = time.monotonic()
    returncode, log_path = Orchestrator._run_subprocess(
        Dummy(),
        "timeout_test",
        [sys.executable, "-c", "import time; time.sleep(10)"],
    )
    elapsed = time.monotonic() - started
    assert returncode == 124
    assert elapsed < 6.0
    assert "stage timeout" in log_path.read_text(encoding="utf-8")
