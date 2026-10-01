"""Rehearse the structural stage on a finished run's frozen targets with the current code.

A full formal run takes hours before ``structure_experiment`` starts. This
reruns only the validation-queue structural step, for the failed frozen
targets of an existing run (or ``--all`` of them), with the code now checked
out, so a repair can be checked before a new formal run is launched.

    python scripts/rehearse_structure_targets.py --run /data/phd_code/runs/<run>
    python scripts/rehearse_structure_targets.py --run <run> --all
    python scripts/rehearse_structure_targets.py --run <run> --targets 6ey6 8h64

* The command is the run's own logged validation-queue command, with only the
  output directory, the target allowlist and the generated-input flags added
  since that run (A45) replaced. The subprocess environment is built by the
  orchestrator's own ``subprocess_environment`` from the run's resolved
  hardware, so OpenMM uses the same platform and precision.
* Everything is written to a new directory outside the run. The run directory
  is only read; reusing it would also reuse its unscreened perturbed inputs.
* The verdict reports physical acceptance, generated-input attempts and
  failure causes only. Recovery RMSDs are written to the rehearsal directory
  but not printed: these targets were already inspected during debugging and
  are historical repair assessments (A45), not a formal result.

This is a pre-launch check. It cannot replace the newly frozen formal run.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from nanoqc.pipeline.orchestrator_common import subprocess_environment  # noqa: E402

LOG_HEADER = re.compile(r"^=== \S+ :: (.+) ===$")


def logged_argv(log_text: str) -> List[str]:
    """The last command the orchestrator logged for a stage subprocess."""
    commands = [match.group(1) for line in log_text.splitlines()
                if (match := LOG_HEADER.match(line.strip()))]
    if not commands:
        raise SystemExit("No logged validation-queue command found")
    return commands[-1].split(" ")


def set_flag(argv: List[str], flag: str, value: str) -> List[str]:
    """Replace a single-valued flag's value, or append the flag."""
    argv = list(argv)
    if flag in argv:
        index = argv.index(flag)
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise SystemExit(f"{flag} has no value in the logged command")
        argv[index + 1] = value
    else:
        argv += [flag, value]
    return argv


def filter_allowlist(entries: Sequence, targets: Sequence[str]) -> list:
    """The frozen allowlist entries for ``targets``, in frozen order."""
    def pdb(entry) -> str:
        if isinstance(entry, str):
            return entry.strip().lower()
        return str(entry.get("target") or entry.get("pdb_id") or "").strip().lower()
    wanted = {target.lower() for target in targets}
    missing = wanted - {pdb(entry) for entry in entries}
    if missing:
        raise SystemExit(f"Not in this run's frozen target list: {sorted(missing)}")
    return [entry for entry in entries if pdb(entry) in wanted]


def summarize(queue: Path, targets: Sequence[str]) -> Dict:
    """Physical acceptance, input attempts and failure causes; no RMSD metrics."""
    summary_path = queue / "run_summary.json"
    run = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    by_method = collections.defaultdict(lambda: [0, 0])
    control_failures, redraws, exhausted = [], [], []
    for path in sorted(glob.glob(str(queue / "results" / "*" / "seed_*" / "experiment"
                                     / "structure_quality_summary.json"))):
        target, seed = Path(path).parts[-4], Path(path).parts[-3]
        for method, outcome in json.loads(Path(path).read_text(encoding="utf-8"))["outcomes"].items():
            passed = outcome["evaluation_status"] == "passed"
            by_method[method][0 if passed else 1] += 1
            if not passed:
                control_failures.append(dict(
                    target=target, seed=seed, method=method,
                    reasons=outcome["evaluation_failure_reasons"],
                    force_rms_kj_mol_nm=outcome.get("movable_force_rms_kj_mol_nm")))
    for path in sorted(glob.glob(str(queue / "results" / "*" / "seed_*" / "perturbation_attempts.json"))):
        ledger = json.loads(Path(path).read_text(encoding="utf-8"))
        where = f"{Path(path).parts[-3]}/{Path(path).parts[-2]}"
        if "selected_attempt" not in ledger:
            exhausted.append(where)
        elif ledger["selected_attempt"] > 0:
            redraws.append((where, ledger["selected_attempt"]))
    failed = sorted((run or {}).get("structure_experiment_failed_targets") or [])
    physical = {failure["target"] for failure in control_failures}
    log_path = queue / "failures.log"
    log = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
    other_causes = {}
    for target in failed:
        if target in physical:
            continue
        lines = next((log[i + 1:i + 60] for i, line in enumerate(log) if line.strip() == target), [])
        last = next((line for line in reversed(lines) if line.strip() and not line.startswith(" ")), "")
        records = sorted(Path(p).name for p in glob.glob(
            str(queue / "results" / target / "**" / "*failure*.json"), recursive=True))
        other_causes[target] = dict(last_log_line=last or None, failure_records=records)
    return dict(
        requested_targets=len(targets),
        closed=(run or {}).get("closed"),
        completed_targets=(run or {}).get("structure_experiment_completed_targets"),
        failed_targets=failed,
        method_pass=dict(sorted((m, f"{p}/{p + q}") for m, (p, q) in by_method.items())),
        physical_failures=control_failures,
        generated_input_redraws=redraws,
        generated_input_exhausted=exhausted,
        failures_without_physical_cause=other_causes,
        ready=bool(run) and not failed and not control_failures and not exhausted,
    )


def report(result: Dict) -> None:
    print(f"\n== 排练结果（{result['requested_targets']} 个目标）")
    print(f"closed={result['closed']}  完成={result['completed_targets']}  失败={result['failed_targets'] or '无'}")
    print("各方法 通过/总数:", result["method_pass"] or "无质量汇总")
    print(f"生成输入：需要重抽的种子 {len(result['generated_input_redraws'])} 个"
          + (f"（例如 {result['generated_input_redraws'][:5]}）" if result["generated_input_redraws"] else "")
          + f"；{len(result['generated_input_exhausted'])} 个种子用完尝试次数仍不合法")
    for failure in result["physical_failures"]:
        force = failure["force_rms_kj_mol_nm"]
        print(f"  物理验收失败 {failure['target']}/{failure['seed']} {failure['method']}: "
              f"{failure['reasons']}" + (f"  RMS 力={force:.3g}" if force is not None else ""))
    for target, cause in result["failures_without_physical_cause"].items():
        print(f"  其他原因失败 {target}: 日志={cause['last_log_line']}  记录={cause['failure_records'] or '无'}")
    print("\n结论:", "可以启动完整 run" if result["ready"] else "先不要启动完整 run，把上面的输出发给我")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="finished run directory to read")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true", help="every frozen validation target")
    scope.add_argument("--targets", nargs="+", help="explicit frozen target PDB IDs")
    parser.add_argument("--out", type=Path, default=None, help="new rehearsal directory (default: beside the run)")
    parser.add_argument("--dry-run", action="store_true", help="print the command and stop")
    args = parser.parse_args(argv)

    run = args.run.resolve()
    queue = run / "validation_queue"
    lock = REPO_ROOT / ".run_full_experiment.lock"
    if lock.is_file():
        # Two jobs on the same GPUs would slow both and could exhaust device memory.
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            alive = True
        except (ValueError, ProcessLookupError):
            alive = False  # stale lock
        except PermissionError:
            alive = True   # exists, owned by another user
        if alive:
            raise SystemExit(f"A formal run is active (PID {pid}); rehearse after it finishes")
    frozen = json.loads((queue / "freeze" / "selected_targets.json").read_text(encoding="utf-8"))
    if args.all:
        targets = filter_allowlist(frozen, [
            str(e if isinstance(e, str) else e.get("target") or e.get("pdb_id")) for e in frozen])
    else:
        names = args.targets or json.loads((queue / "run_summary.json").read_text(
            encoding="utf-8"))["structure_experiment_failed_targets"]
        if not names:
            raise SystemExit("The run has no failed structural targets; pass --all to rehearse every target")
        targets = filter_allowlist(frozen, names)
    names = [str(e if isinstance(e, str) else e.get("target") or e.get("pdb_id")).lower() for e in targets]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = (args.out or run.parent / f"rehearsal_{run.name}_{stamp}").resolve()
    if run == out or run in out.parents:
        raise SystemExit("The rehearsal directory must be outside the run directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out")

    config = yaml.safe_load((run / "frozen_config.yaml").read_text(encoding="utf-8"))
    current = yaml.safe_load((REPO_ROOT / "configs" / "full_experiment_config.yaml").read_text(encoding="utf-8"))
    command = logged_argv((run / "logs" / "structure_experiment_validation_queue.log").read_text(encoding="utf-8"))
    allowlist = out / "selected_targets.json"
    command = set_flag(command, "--out-dir", str(out / "validation_queue"))
    command = set_flag(command, "--pdb-allowlist-file", str(allowlist))
    # Generated-input flags added after older runs (A45); values from the current config.
    command = set_flag(command, "--perturbation-max-attempts", str(
        (current.get("structure_experiment") or {}).get("perturbation_max_attempts", 32)))
    command = set_flag(command, "--min-input-heavy-distance", str(
        (current.get("data_audit") or {}).get("min_interresidue_heavy_distance_angstrom", 1.0)))
    env = subprocess_environment(config.get("hardware", {}), REPO_ROOT)

    print(f"排练 {len(names)} 个目标: {names}")
    print(f"输出目录: {out}")
    print(f"OpenMM: {env['QP_OPENMM_PLATFORM']} / {env['QP_OPENMM_PRECISION']}")
    if args.dry_run:
        print("命令:", " ".join(command))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    allowlist.write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
    (out / "rehearsal.json").write_text(json.dumps(dict(
        source_run=str(run), targets=names, command=command,
        purpose="pre-launch repair rehearsal on already-inspected targets; not a formal result",
    ), indent=2) + "\n", encoding="utf-8")
    log = out / "rehearsal.log"
    print(f"运行中，日志: {log}")
    with log.open("w", encoding="utf-8") as handle:
        returncode = subprocess.run(command, cwd=str(REPO_ROOT), env=env,
                                    stdout=handle, stderr=subprocess.STDOUT).returncode
    print(f"子进程退出码: {returncode}")
    result = summarize(out / "validation_queue", names)
    (out / "rehearsal_verdict.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report(result)
    return 0 if result["ready"] and returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
