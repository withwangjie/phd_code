from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoqc.pipeline.run_full_experiment import Orchestrator, calibration_solver_args
from nanoqc.reporting.generate_final_research_report import training_coarse_atomistic_rank_diagnostic


def test_diagnostic_calibration_is_never_passed_to_solvers(tmp_path: Path) -> None:
    fit = tmp_path / "fit.json"
    fit.write_text("{}", encoding="utf-8")
    diagnostic = {"mode": "diagnostic", "require_calibrated": False}
    assert calibration_solver_args(diagnostic, fit) == []
    assert calibration_solver_args(diagnostic, fit, force_required=True) == []
    assert calibration_solver_args({"mode": "frozen", "require_calibrated": True}, fit) == [
        "--require-calibrated-energy", "--energy-calibration-file", str(fit)
    ]


def test_training_assignment_rank_diagnostic_keeps_nonestimable_groups() -> None:
    def row(pdb: str, coarse: float, amber: float) -> dict[str, str]:
        return dict(pdb_id=pdb, split="train", prior_energy=str(coarse),
                    vhh_environment_energy="0", antigen_energy="0", pair_energy="0",
                    amber_delta_kcal=str(amber))
    result=training_coarse_atomistic_rank_diagnostic([
        row("1ABC",0,2),row("1ABC",1,1),row("1ABC",2,0),
        row("2DEF",0,0),row("2DEF",0,1),row("2DEF",0,2),
    ])
    assert result["n_pdb"]==2
    assert result["n_estimable"]==1
    assert result["n_nonestimable"]==1
    assert result["median_within_pdb_spearman"]==-1.0
    assert result["per_pdb"][1]["spearman_rho"] is None
    held_out=row("3GHI",0,0)
    held_out["split"]="test_snac_hard"
    with pytest.raises(ValueError, match="training rows only"):
        training_coarse_atomistic_rank_diagnostic([held_out])


def test_failed_amber_fit_acceptance_remains_auditable_diagnostic(tmp_path: Path) -> None:
    class Harness:
        run_dir = tmp_path
        venv_python = "python"
        config = {
            "paths": {"repo_root": str(tmp_path), "data_root": str(tmp_path / "data")},
            "master_seed": 4050350448,
            "queue_freeze": {"homology_isolation": {}},
            "structure_experiment": {"solvent_model": "vacuum"},
            "qc_benchmark": {
                "checkpoint": "unused.pt",
                "rotamer_model": {"mode": "pyrosetta_dun10"},
                "energy_calibration": {
                    "mode": "diagnostic", "selection_mode": "contact",
                    "acceptance": {
                        "min_eligible_complexes": 2,
                        "min_train_complexes": 2,
                        "min_train_groups": 2,
                        "min_cv_folds": 2,
                        "max_cv_rmse_kcal": 10.0,
                        "max_generation_failure_fraction": 0.1,
                        "require_family_grouped_cv": True,
                    },
                },
            },
        }

        def dataset_dir(self):
            return tmp_path / "dataset"

        def checkpoint_dir(self):
            return tmp_path / "checkpoints"

        def frozen_cluster_map_path(self):
            return None

        def _run_subprocess(self, name, argv):
            root = tmp_path / "calibration"
            root.mkdir(exist_ok=True)
            if name == "energy_calibration_dataset":
                (root / "coarse_to_amber_train.csv").write_text(
                    "pdb_id,split\n1abc,train\n2def,train\n", encoding="utf-8"
                )
                (root / "coarse_to_amber_train.provenance.json").write_text(json.dumps({
                    "complexes_discovered": 2, "complexes_attempted": 2,
                    "complexes_succeeded": 2, "input_quality_exclusions": [],
                    "input_quality_exclusion_fraction": 0.0, "failures": [],
                    "generation_failure_fraction": 0.0,
                }), encoding="utf-8")
            else:
                assert name == "energy_calibration_fit"
                (root / "coarse_to_amber.json").write_text(json.dumps({
                    "n_train_complexes": 2, "n_train_groups": 2,
                    "cv_fold_count": 2, "cv_grouping": "family_cluster",
                    "cv_rmse_kcal": 2.64e19,
                }), encoding="utf-8")
            return 0, root / f"{name}.log"

    harness = Harness()
    result = Orchestrator.stage_energy_calibration(harness)
    assert result.status == "completed_with_failures"
    assessment = json.loads((tmp_path / "calibration" / "diagnostic_assessment.json").read_text())
    assert assessment["accepted"] is False
    assert assessment["applied_to_solver"] is False
    assert any("cv_rmse_kcal" in reason for reason in assessment["failure_reasons"])
    harness._artifacts_present = Orchestrator._artifacts_present
    ok, detail = Orchestrator._validate_completed_stage_artifacts(
        harness, "energy_calibration", require_results_manifest=False
    )
    assert ok, detail

    original_run = harness._run_subprocess

    def interrupted_generation(name, argv):
        assert name == "energy_calibration_dataset"
        return 1, tmp_path / "calibration" / "resource_failure.log"

    harness._run_subprocess = interrupted_generation
    result = Orchestrator.stage_energy_calibration(harness)
    assert result.status == "failed"
    assert not result.artifacts_ok

    def failed_fit(name, argv):
        if name == "energy_calibration_fit":
            return 1, tmp_path / "calibration" / "failed_fit.log"
        return original_run(name, argv)

    harness._run_subprocess = failed_fit
    result = Orchestrator.stage_energy_calibration(harness)
    assert result.status == "completed_with_failures"
    assessment = json.loads((tmp_path / "calibration" / "diagnostic_assessment.json").read_text())
    assert assessment["fit_status"] == "failed"
    assert assessment["accepted"] is False
    ok, detail = Orchestrator._validate_completed_stage_artifacts(
        harness, "energy_calibration", require_results_manifest=False
    )
    assert ok, detail
