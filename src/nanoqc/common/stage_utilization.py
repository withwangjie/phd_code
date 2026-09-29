"""Summarize per-stage CPU and GPU utilization of a run (read-only).

usage: python -m nanoqc.common.stage_utilization RUN_DIR [--target 80] [--json OUT]

Reads ``logs/<stage>.cpu.csv`` (CPUStageMonitor) and ``logs/<stage>.gpu.csv``
(GPUStageMonitor). For each stage it reports the sampled wall time, the mean
CPU share of the stage's own process tree and of the whole host (percent of
logical CPUs), and each GPU's mean utilization and peak memory. A stage is
marked ``below`` when its stage CPU share and every GPU mean stay under the
target. Mean utilization is not throughput: compare completed work per hour
too (see docs/PIPELINE_GPU_UTILIZATION.md).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "sample"]


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 1) if values else None


def _span_seconds(rows: List[Dict[str, str]]) -> Optional[float]:
    stamps = sorted(datetime.fromisoformat(row["utc"]) for row in rows if row.get("utc"))
    return round((stamps[-1] - stamps[0]).total_seconds(), 1) if len(stamps) > 1 else None


def summarize(run_dir: Path, target: float = 80.0) -> List[Dict[str, Any]]:
    log_dir = Path(run_dir) / "logs"
    stages = sorted({p.name.split(".")[0] for p in log_dir.glob("*.cpu.csv")}
                    | {p.name.split(".")[0] for p in log_dir.glob("*.gpu.csv")})
    summary = []
    for stage in stages:
        entry: Dict[str, Any] = {"stage": stage}
        cpu_path, gpu_path = log_dir / f"{stage}.cpu.csv", log_dir / f"{stage}.gpu.csv"
        cpu = _rows(cpu_path) if cpu_path.is_file() else []
        entry["cpu_samples"] = len(cpu)
        entry["sampled_seconds"] = _span_seconds(cpu)
        entry["stage_cpu_percent_mean"] = _mean([v for r in cpu if (v := _float(r["stage_cpu_percent"])) is not None])
        entry["host_cpu_percent_mean"] = _mean([v for r in cpu if (v := _float(r["host_cpu_percent"])) is not None])
        entry["stage_rss_gib_max"] = max([v for r in cpu if (v := _float(r.get("stage_rss_gib"))) is not None],
                                         default=None)
        gpu = _rows(gpu_path) if gpu_path.is_file() else []
        by_device: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in gpu:
            by_device[row["gpu_index"]].append(row)
        entry["gpus"] = {
            index: {"utilization_percent_mean": _mean([v for r in rows if (v := _float(r["utilization_percent"])) is not None]),
                    "memory_used_mib_max": max([v for r in rows if (v := _float(r["memory_used_mib"])) is not None],
                                               default=None)}
            for index, rows in sorted(by_device.items())}
        levels = [entry["stage_cpu_percent_mean"]] + [g["utilization_percent_mean"] for g in entry["gpus"].values()]
        levels = [v for v in levels if v is not None]
        entry["status"] = "no samples" if not levels else ("ok" if max(levels) >= target else "below")
        summary.append(entry)
    return summary


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--target", type=float, default=80.0, help="utilization target in percent (default 80)")
    parser.add_argument("--json", type=Path, default=None, help="also write the summary as JSON here")
    args = parser.parse_args(argv)
    summary = summarize(args.run_dir, args.target)
    if not summary:
        print(f"No stage telemetry under {args.run_dir / 'logs'} (enable cpu/gpu_monitor_enabled).")
        return 1
    print(f"| Stage | Sampled s | Stage CPU % | Host CPU % | Peak RSS GiB | GPU mean % (peak MiB) | vs {args.target:g}% |")
    print("|---|---:|---:|---:|---:|---|---|")
    for e in summary:
        gpus = ", ".join(f"GPU{i}: {_fmt(g['utilization_percent_mean'])} ({_fmt(g['memory_used_mib_max'])})"
                         for i, g in e["gpus"].items()) or "-"
        print(f"| {e['stage']} | {_fmt(e['sampled_seconds'])} | {_fmt(e['stage_cpu_percent_mean'])} | "
              f"{_fmt(e['host_cpu_percent_mean'])} | {_fmt(e['stage_rss_gib_max'])} | {gpus} | {e['status']} |")
    if args.json:
        args.json.write_text(json.dumps({"target_percent": args.target, "stages": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
