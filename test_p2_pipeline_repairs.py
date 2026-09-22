"""Regression tests for P2 pipeline hardening."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import build_final_pyg_dataset as bfp
import continue_server_pipeline as legacy
from run_full_experiment import Orchestrator, StageResult


def test_singleton_hard_set_pair_similarity_is_zero() -> None:
    assert bfp.max_pair_similarity([]) == 0.0
    assert bfp.max_pair_similarity(["CARDRST"]) == 0.0


def test_dev_subruns_use_single_target_and_emit_root_summary() -> None:
    source = inspect.getsource(Orchestrator.stage_structure_experiment)
    assert '"--targets", "1"' in source
    assert 'out_dir / "run_summary.json"' in source
    assert 'planned_target_ids=sorted(planned_dev)' in source
    assert 'structure_experiment_completed_target_ids=sorted(dev_completed)' in source


class _StageHarness:
    only = None
    force_restage = set()

    def __init__(self, records):
        self.records = dict(records)
        self.saved = []
        self.config = {"stages": {"target": True}}

    def _load_stage_status(self, stage):
        return self.records.get(stage)

    def _save_stage_status(self, result):
        self.saved.append(result)


def _completed_target() -> StageResult:
    return StageResult(
        "target", "completed", "start", "finish", 0, "ok",
    )


def test_disabled_required_prerequisite_does_not_satisfy_dependency() -> None:
    harness = _StageHarness({"data_audit": {"status": "skipped"}})
    result = Orchestrator.run_stage(
        harness, "target", ["data_audit"], _completed_target,
    )
    assert result.status == "failed"
    assert "has not completed" in result.detail


def test_smoke_skip_is_explicitly_optional() -> None:
    harness = _StageHarness({"smoke_check": {"status": "skipped"}})
    result = Orchestrator.run_stage(
        harness, "target", ["smoke_check"], _completed_target,
    )
    assert result.status == "completed"


def test_legacy_pipeline_requires_explicit_acknowledgement() -> None:
    with pytest.raises(SystemExit) as exc:
        legacy.main(["123"])
    assert exc.value.code == 2
