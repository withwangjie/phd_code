"""Regression tests for final formal-run gates."""
from __future__ import annotations

import inspect
from pathlib import Path

import batch_benchmark_hard_set as bbh
import run_real_complex_pilot as rcp
from run_full_experiment import Orchestrator, apply_runtime_mode_overrides


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



def test_smoke_only_override_enables_smoke_without_mutating_source_config() -> None:
    original = {"stages": {"env_check": True, "smoke_check": False}}
    resolved = apply_runtime_mode_overrides(original, smoke_only=True)
    assert resolved["stages"]["env_check"] is True
    assert resolved["stages"]["smoke_check"] is True
    assert original["stages"]["smoke_check"] is False


def test_structural_perturbation_range_is_explicitly_propagated() -> None:
    pilot_source = inspect.getsource(rcp.main)
    orchestrator_source = inspect.getsource(Orchestrator.stage_structure_experiment)
    assert "--min-perturb-degrees" in pilot_source
    assert "--max-perturb-degrees" in pilot_source
    assert '"--min-perturb-degrees", str(cfg.get("min_perturb_degrees", 40.0))' in orchestrator_source
    assert '"--max-perturb-degrees", str(cfg.get("max_perturb_degrees", 120.0))' in orchestrator_source


def test_qaoa_output_curve_reuses_one_optimization_per_variant() -> None:
    source = inspect.getsource(bbh._ablation_run_case)
    assert "qaoa_cache = {}" in source
    assert "if cache_key not in qaoa_cache:" in source
    assert "qaoa_cache[cache_key] = (opt, optimization, optimization_seconds)" in source
    assert "elapsed = optimization_seconds + sampling_seconds" in source


def test_resume_qc_marker_requires_closed_summary(tmp_path: Path) -> None:
    class Dummy:
        run_dir = tmp_path
        config = {"statistics": {}, "final_report": {}}

        def dataset_dir(self):
            return tmp_path / "dataset"

        def checkpoint_dir(self):
            return tmp_path / "checkpoints"

        _artifacts_present = staticmethod(Orchestrator._artifacts_present)

    root = tmp_path / "qc_benchmark"
    root.mkdir()
    (root / "run_manifest.json").write_text("{}", encoding="utf-8")
    (root / "run_summary.json").write_text(
        '{"closed": false, "cases_completed_total": 1}', encoding="utf-8")
    ok, detail = Orchestrator._validate_completed_stage_artifacts(Dummy(), "qc_benchmark")
    assert ok is False
    assert "not closed" in detail



def test_completed_prerequisite_with_missing_artifacts_blocks_target_stage() -> None:
    class Harness:
        only = None
        force_restage = set()
        config = {"stages": {"target": True}}

        def __init__(self):
            self.saved = []

        def _load_stage_status(self, stage):
            if stage == "qc_benchmark":
                return {"status": "completed"}
            return None

        def _save_stage_status(self, result):
            self.saved.append(result)

        def _validate_completed_stage_artifacts(self, stage):
            assert stage == "qc_benchmark"
            return False, "run_summary.json missing"

    result = Orchestrator.run_stage(
        Harness(), "target", ["qc_benchmark"],
        lambda: None,
    )
    assert result.status == "failed"
    assert "failed artifact verification" in result.detail
