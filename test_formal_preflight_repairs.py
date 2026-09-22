"""Regression tests for final formal-run gates."""
from __future__ import annotations

from pathlib import Path

from run_full_experiment import Orchestrator


class _StatsHarness:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.venv_python = "python"
        self.config = {
            "master_seed": 20260917,
            "qc_benchmark": {
                "outputs": [10, 30, 100, 300, 1000],
                "qaoa_objective": ["mean", "cvar"],
                "qaoa_restarts": [1, 4],
            },
            "statistics": {
                "resamples": 100,
                "budget_modes": ["outputs", "time"],
                "primary_outputs": 1000,
                "primary_objective": "cvar",
                "primary_restarts": 4,
                "max_time_overrun_fraction": 0.10,
                "cluster_map": None,
            },
        }

    def _run_subprocess(self, stage, argv):
        return 0, self.run_dir / f"{stage}.log"


def test_statistics_fails_closed_when_qc_results_dir_missing(tmp_path: Path) -> None:
    harness = _StatsHarness(tmp_path)
    result = Orchestrator.stage_statistics(harness)
    assert result.status == "failed"
    assert result.returncode == 1
    assert result.artifacts_ok is False
    assert "input directory is missing" in result.detail


def test_statistics_fails_closed_when_expected_artifacts_missing(tmp_path: Path) -> None:
    (tmp_path / "qc_benchmark").mkdir()
    harness = _StatsHarness(tmp_path)
    result = Orchestrator.stage_statistics(harness)
    assert result.status == "failed"
    assert result.returncode == 1
    assert result.artifacts_ok is False
    assert "required statistics artifacts" in result.detail


def test_statistics_fails_closed_on_primary_contrast_config_drift(tmp_path: Path) -> None:
    (tmp_path / "qc_benchmark").mkdir()
    harness = _StatsHarness(tmp_path)
    harness.config["statistics"]["primary_outputs"] = 999
    result = Orchestrator.stage_statistics(harness)
    assert result.status == "failed"
    assert "primary_outputs=999" in result.detail
