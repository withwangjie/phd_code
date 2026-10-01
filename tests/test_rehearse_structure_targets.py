"""Pre-launch structural rehearsal: same command, same environment, separate output."""
import inspect
import json
from pathlib import Path

import pytest

from scripts import rehearse_structure_targets as rehearse
from nanoqc.pipeline import run_full_experiment
from nanoqc.pipeline.orchestrator_common import subprocess_environment


def test_rehearsal_and_orchestrator_share_one_environment_builder():
    """OpenMM platform/precision must not fall back to Reference in a rehearsal."""
    source = inspect.getsource(run_full_experiment.Orchestrator._run_subprocess)
    assert "subprocess_environment(hardware" in source
    hardware = dict(cpu_threads_per_process=1, openmm_cpu_threads=8, openmm_platform="CUDA",
                    openmm_device="0", openmm_precision="double")
    env = subprocess_environment(hardware, Path("/repo"), base={"PYTHONPATH": "/x"})
    assert env["QP_OPENMM_PLATFORM"] == "CUDA" and env["QP_OPENMM_PRECISION"] == "double"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["PYTHONPATH"].split(":")[0] == "/repo/src" and env["PYTHONPATH"].endswith("/x")


def test_logged_command_is_reused_with_only_the_rehearsal_flags_replaced():
    log = ("starting\n=== 2026-09-30T01:00:00Z :: /v/python -m nanoqc.x --out-dir /run/old "
           "--pdb-allowlist-file /run/old/freeze/selected_targets.json --sites 6 ===\nnoise\n"
           "=== 2026-09-30T02:00:00Z :: /v/python -m nanoqc.x --out-dir /run/vq "
           "--pdb-allowlist-file /run/vq/freeze/selected_targets.json --sites 6 ===\n")
    argv = rehearse.logged_argv(log)
    assert argv[0] == "/v/python" and argv[-2:] == ["--sites", "6"]
    argv = rehearse.set_flag(argv, "--out-dir", "/new/vq")
    argv = rehearse.set_flag(argv, "--pdb-allowlist-file", "/new/list.json")
    argv = rehearse.set_flag(argv, "--perturbation-max-attempts", "32")
    assert argv[argv.index("--out-dir") + 1] == "/new/vq"
    assert argv[argv.index("--pdb-allowlist-file") + 1] == "/new/list.json"
    assert argv[-2:] == ["--perturbation-max-attempts", "32"]
    assert "/run/vq" not in " ".join(argv)
    with pytest.raises(SystemExit):
        rehearse.logged_argv("no command here")


def test_allowlist_filter_keeps_frozen_entries_and_order():
    frozen = ["1ABC", dict(target="2def", eligibility_compatible_residues=["H:31"]), dict(pdb_id="3GHI")]
    assert rehearse.filter_allowlist(frozen, ["3ghi", "1abc"]) == ["1ABC", dict(pdb_id="3GHI")]
    assert rehearse.filter_allowlist(frozen, ["2DEF"]) == [frozen[1]]
    with pytest.raises(SystemExit, match="Not in this run's frozen target list"):
        rehearse.filter_allowlist(frozen, ["9zzz"])


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_verdict_reports_physics_and_causes_without_recovery_metrics(tmp_path):
    queue = tmp_path / "validation_queue"
    _write(queue / "run_summary.json", dict(closed=True, structure_experiment_completed_targets=1,
                                            structure_experiment_failed_targets=["8h64", "9xyz"]))
    ok = dict(evaluation_status="passed", evaluation_failure_reasons=[], movable_force_rms_kj_mol_nm=2.0)
    bad = dict(evaluation_status="failed", evaluation_failure_reasons=["relaxation_not_converged"],
               movable_force_rms_kj_mol_nm=3.7e8)
    _write(queue / "results/6ey6/seed_44/experiment/structure_quality_summary.json",
           dict(outcomes=dict(qaoa=ok, relax_only=ok)))
    _write(queue / "results/8h64/seed_45/experiment/structure_quality_summary.json",
           dict(outcomes=dict(qaoa=ok, relax_only=bad)))
    _write(queue / "results/6ey6/seed_44/perturbation_attempts.json", dict(selected_attempt=0))
    _write(queue / "results/8h64/seed_45/perturbation_attempts.json", dict(selected_attempt=3))
    _write(queue / "results/9xyz/seed_42/perturbation_attempts.json", dict(seed=42, attempts=[]))
    (queue / "failures.log").write_text("9xyz\nTraceback (most recent call last):\n  File x\n"
                                        "ValueError: Test graph hash mismatch\n", encoding="utf-8")
    result = rehearse.summarize(queue, ["6ey6", "8h64", "9xyz"])
    assert result["method_pass"] == {"qaoa": "2/2", "relax_only": "1/2"}
    assert result["physical_failures"][0]["target"] == "8h64"
    assert result["generated_input_redraws"] == [("8h64/seed_45", 3)]
    assert result["generated_input_exhausted"] == ["9xyz/seed_42"]
    assert result["failures_without_physical_cause"]["9xyz"]["last_log_line"] == \
        "ValueError: Test graph hash mismatch"
    assert result["ready"] is False
    assert "rmsd" not in json.dumps(result).lower()


def test_rehearsal_never_writes_inside_the_run(tmp_path):
    run = tmp_path / "run"
    (run / "validation_queue" / "freeze").mkdir(parents=True)
    (run / "validation_queue" / "freeze" / "selected_targets.json").write_text('["6ey6"]')
    with pytest.raises(SystemExit, match="outside the run directory"):
        rehearse.main(["--run", str(run), "--all", "--out", str(run / "rehearsal")])


def test_dry_run_wires_the_failed_targets_command_and_cuda_environment(tmp_path, capsys):
    run = tmp_path / "experiments_full_run_x"
    queue = run / "validation_queue"
    _write(queue / "freeze" / "selected_targets.json", ["6ey6", dict(target="8h64"), "1abc"])
    _write(queue / "run_summary.json", dict(structure_experiment_failed_targets=["8h64"]))
    (run / "frozen_config.yaml").write_text(
        "hardware:\n  openmm_platform: CUDA\n  openmm_precision: double\n  cpu_threads_per_process: 1\n",
        encoding="utf-8")
    (run / "logs").mkdir()
    (run / "logs" / "structure_experiment_validation_queue.log").write_text(
        f"=== 2026-09-30T01:00:00Z :: /v/python -m nanoqc.experiments.run_real_complex_pilot "
        f"--out-dir {queue} --pdb-allowlist-file {queue}/freeze/selected_targets.json --sites 6 ===\n",
        encoding="utf-8")
    out = tmp_path / "rehearsal"
    assert rehearse.main(["--run", str(run), "--out", str(out), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "['8h64']" in printed and "CUDA / double" in printed
    assert f"--out-dir {out}/validation_queue" in printed
    assert "--perturbation-max-attempts 32" in printed and "--min-input-heavy-distance 1.0" in printed
    assert not out.exists(), "a dry run writes nothing"
