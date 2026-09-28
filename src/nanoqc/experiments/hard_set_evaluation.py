"""End-to-end XY-mixer QAOA versus SA benchmark on the SNAC hard set.

The pipeline loads each PyG complex, ranks and prunes its VHH interface with
the EGNN scorer, builds a penalty-free physical rotamer QUBO, enumerates the
exact feasible optimum, and benchmarks XY-mixer QAOA against simulated
annealing.  Per-target failures are recorded in the output CSV and never abort
the remaining batch.

For the 80-vCPU server, CPU simulation defaults to 20 spawned workers with
three native threads each. ``lightning.gpu`` caps concurrency to
``gpu_count * gpu_workers_per_device`` and assigns workers round-robin through
``CUDA_VISIBLE_DEVICES``.
"""
# This docstring is also the default mode's --help text (_build_parser passes
# description=__doc__); keep it identical to what the benchmark always printed.

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import gc
import math
import multiprocessing as mp
import os
import sys
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pennylane as qml
import torch
from nanoqc.data.safe_graph_load import load_graph
from tqdm.auto import tqdm
from nanoqc.model.model_egnn_pruning import EGNNInterfaceScorer, extract_top_interface_subgraph
from nanoqc.model.model_egnn_pruning import (  # re-exported: historical import location
    ModelLoadInfo, _clean_error_message, load_interface_scorer,
)
from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler
from nanoqc.qubo.subgraph_to_qubo import InterfaceQUBOBuilder
from nanoqc.common.repo_io import repo_path
from nanoqc.experiments.benchmark_common import SHARED_HELPER_MODULES



CSV_FILENAME = "snac_hard_qaoa_vs_sa_metrics.csv"
REPORT_FILENAME = "benchmark_summary_report.md"
FAILED_LOG_FILENAME = "failed_cases.log"
OPTIMIZATION_WARNINGS_FILENAME = "optimization_warnings.log"

_WORKER_SCORER: Optional[EGNNInterfaceScorer] = None
_WORKER_ARGS: Optional[argparse.Namespace] = None

CSV_FIELDS = (
    "status",
    "simulation_mode",
    "target_id",
    "pdb_id",
    "cdr3_len",
    "source_file",
    "num_input_nodes",
    "num_subgraph_nodes",
    "num_frozen_residues",
    "num_selected_sites",
    "num_bits",
    "configuration_count",
    "ground_degeneracy",
    "e_ground",
    "e_qaoa",
    "qaoa_gap",
    "qaoa_hit_ground",
    "qaoa_success_probability",
    "qaoa_legal_rate",
    "qaoa_conformational_entropy",
    "qaoa_low_energy_coverage",
    "qaoa_low_energy_unique_sampled",
    "qaoa_unique_samples",
    "qaoa_expected_energy",
    "optimization_energy_start",
    "optimization_energy_end",
    "optimization_energy_drop",
    "optimization_steps_converged",
    "optimization_warning",
    "qaoa_optimizer_success",
    "qaoa_optimizer_evaluations",
    "qaoa_seconds",
    "qaoa_optimize_seconds",
    "qaoa_sample_seconds",
    "e_sa",
    "sa_gap",
    "sa_hit_ground",
    "sa_success_probability",
    "sa_legal_rate",
    "sa_conformational_entropy",
    "sa_low_energy_coverage",
    "sa_low_energy_unique_sampled",
    "sa_unique_samples",
    "low_energy_state_count",
    "sa_seconds",
    "load_seconds",
    "pruning_seconds",
    "qubo_seconds",
    "ground_truth_seconds",
    "total_seconds",
    "error_stage",
    "error_type",
    "error_message",
)


class TargetEvaluationError(RuntimeError):
    """Exception enriched with the pipeline stage that failed."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def _scalar(value: Any, default: Any = "") -> Any:
    """Convert common tensor/list scalar attributes into CSV-safe values."""

    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return str(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar(value[0], default)
    return value


def _zero_small_gap(value: float, tolerance: float = 1e-9) -> float:
    """Remove harmless floating-point noise from a nonnegative energy gap."""

    return 0.0 if abs(value) <= tolerance else float(value)


def _sample_diversity(
    counts: Mapping[Tuple[int, ...], int],
    low_energy_states: Sequence[Tuple[int, ...]],
) -> Tuple[float, float, int, int]:
    """Return Shannon entropy (nats) and exact low-energy ensemble coverage."""

    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("Sample counts must contain at least one observation")
    probabilities = np.asarray(
        [value / total for value in counts.values() if value > 0], dtype=np.float64
    )
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    low_energy = set(low_energy_states)
    sampled_low = low_energy.intersection(counts)
    coverage = len(sampled_low) / len(low_energy) if low_energy else 0.0
    return entropy, float(coverage), len(sampled_low), len(counts)


def evaluate_single_target(
    graph_path: Path,
    scorer: EGNNInterfaceScorer,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run pruning, QUBO construction, exact truth, QAOA, and SA for one graph.

    Args:
        graph_path: Path to one serialized PyG ``Data`` object.
        scorer: Loaded or deterministically initialized EGNN interface scorer.
        args: Parsed command-line configuration.

    Returns:
        One flat dictionary conforming to ``CSV_FIELDS``.

    Raises:
        TargetEvaluationError: Wraps any failure with the responsible stage.
    """

    target_start = time.perf_counter()
    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "status": "success",
            "target_id": graph_path.stem,
            "source_file": str(graph_path.resolve()),
        }
    )
    stage = "load"
    print(f"START {graph_path.name} pid={os.getpid()}", flush=True)
    try:
        start = time.perf_counter()
        data = load_graph(graph_path)
        row["load_seconds"] = time.perf_counter() - start
        row["pdb_id"] = str(_scalar(getattr(data, "pdb_id", graph_path.stem)))
        cdr3_len = _scalar(getattr(data, "cdr3_len", ""))
        row["cdr3_len"] = "" if cdr3_len == "" else int(cdr3_len)
        row["num_input_nodes"] = int(data.num_nodes)

        stage = "egnn_pruning"
        start = time.perf_counter()
        subgraph = extract_top_interface_subgraph(
            data,
            scorer,
            probability_threshold=0.5,
            min_active=5,
            max_active=10,
            environment_radius=6.0,
        )
        row["pruning_seconds"] = time.perf_counter() - start
        row["num_subgraph_nodes"] = int(subgraph.num_nodes)
        selected_mask = getattr(subgraph, "is_active", None)
        row["num_selected_sites"] = (
            int(selected_mask.sum().item()) if selected_mask is not None else ""
        )
        frozen_mask = getattr(subgraph, "is_frozen_environment", None)
        row["num_frozen_residues"] = (
            int(frozen_mask.sum().item()) if frozen_mask is not None else ""
        )

        stage = "qubo_build"
        start = time.perf_counter()
        active_count = int(selected_mask.sum().item())
        builder = InterfaceQUBOBuilder(
            min_variables=min(20, 3 * active_count),
            max_variables=30,
            max_sites=10,
        )
        qubo_res = builder.build(subgraph)
        row["qubo_seconds"] = time.perf_counter() - start
        row["num_bits"] = int(len(qubo_res.physical_self))
        row["num_selected_sites"] = int(len(qubo_res.site_to_variables))

        stage = "sampler_initialization"
        sampler = XYMixerQAOASampler(
            qubo_res.physical_self,
            qubo_res.physical_pair,
            qubo_res.site_to_variables,
            p=2,
            shots=args.shots,
            device_name=args.backend,
            seed=42,
            simulation_mode=args.simulation_mode,
        )
        row["simulation_mode"] = args.simulation_mode
        print(f"QAOA {graph_path.name}: bits={sampler.num_variables}, legal_states={sampler.feasible_configuration_count}, mode={args.simulation_mode}", flush=True)

        stage = "exact_ground_state"
        start = time.perf_counter()
        ground = sampler.enumerate_ground_states()
        row["ground_truth_seconds"] = time.perf_counter() - start
        row["configuration_count"] = int(ground.configuration_count)
        row["ground_degeneracy"] = int(len(ground.states))
        row["e_ground"] = float(ground.energy)
        low_energy_states = sampler.low_energy_states(ground.energy + 2.0)
        row["low_energy_state_count"] = len(low_energy_states)

        stage = "qaoa_optimization"
        qaoa_start = time.perf_counter()
        start = time.perf_counter()
        def progress(evaluation: int, energy: float) -> None:
            if evaluation == 1 or evaluation % 5 == 0:
                print(f"OPT {graph_path.name} eval={evaluation}/{args.qaoa_max_evals} energy={energy:.8g} elapsed={time.perf_counter()-qaoa_start:.1f}s", flush=True)
            if time.perf_counter() - qaoa_start > args.optimization_timeout:
                raise TimeoutError("QAOA optimization exceeded time budget between evaluations")
        opt_res = sampler.optimize(
            method=args.qaoa_optimizer, max_iterations=args.qaoa_max_evals,
            progress_callback=progress,
        )
        row["qaoa_optimize_seconds"] = time.perf_counter() - start
        row["qaoa_expected_energy"] = float(opt_res.energy)
        energy_start = float(opt_res.history[0]) if opt_res.history else float(opt_res.energy)
        energy_end = float(opt_res.energy)
        energy_drop = energy_start - energy_end
        row["optimization_energy_start"] = energy_start
        row["optimization_energy_end"] = energy_end
        row["optimization_energy_drop"] = energy_drop
        row["optimization_steps_converged"] = int(opt_res.evaluations) if opt_res.success else ""
        warning_threshold = max(1e-6, 1e-3 * max(abs(energy_start), 1.0))
        if energy_drop <= warning_threshold:
            row["optimization_warning"] = (
                f"small energy decrease ({energy_drop:.3e} <= {warning_threshold:.3e})"
            )
        row["qaoa_optimizer_success"] = int(bool(opt_res.success))
        if not opt_res.success:
            row["optimization_warning"] = (str(row["optimization_warning"]) + "; optimizer: " + opt_res.message).lstrip("; ")
        row["qaoa_optimizer_evaluations"] = int(opt_res.evaluations)

        stage = "qaoa_sampling"
        start = time.perf_counter()
        qaoa_res = sampler.sample(
            opt_res,
            shots=args.shots,
            ground_state=ground,
        )
        row["qaoa_sample_seconds"] = time.perf_counter() - start
        row["qaoa_seconds"] = time.perf_counter() - qaoa_start
        qaoa_gap = float(qaoa_res.best_energy - ground.energy)
        row["e_qaoa"] = float(qaoa_res.best_energy)
        row["qaoa_gap"] = _zero_small_gap(qaoa_gap)
        row["qaoa_hit_ground"] = int(
            np.isclose(qaoa_res.best_energy, ground.energy, atol=1e-6, rtol=0.0)
        )
        row["qaoa_success_probability"] = float(
            qaoa_res.ground_state_success_probability
        )
        row["qaoa_legal_rate"] = float(qaoa_res.legal_rate)
        (
            row["qaoa_conformational_entropy"],
            row["qaoa_low_energy_coverage"],
            row["qaoa_low_energy_unique_sampled"],
            row["qaoa_unique_samples"],
        ) = _sample_diversity(qaoa_res.counts, low_energy_states)

        stage = "simulated_annealing"
        start = time.perf_counter()
        sa_res = sampler.simulated_annealing(
            num_reads=args.sa_reads,
            sweeps=args.sa_sweeps,
            ground_state=ground,
        )
        row["sa_seconds"] = time.perf_counter() - start
        if not sampler.is_legal(sa_res.best_state):
            raise AssertionError("SA returned an illegal one-hot assignment")
        sa_gap = float(sa_res.best_energy - ground.energy)
        row["e_sa"] = float(sa_res.best_energy)
        row["sa_gap"] = _zero_small_gap(sa_gap)
        row["sa_hit_ground"] = int(
            np.isclose(sa_res.best_energy, ground.energy, atol=1e-6, rtol=0.0)
        )
        row["sa_success_probability"] = float(
            sa_res.ground_state_success_probability
        )
        row["sa_legal_rate"] = 1.0
        (
            row["sa_conformational_entropy"],
            row["sa_low_energy_coverage"],
            row["sa_low_energy_unique_sampled"],
            row["sa_unique_samples"],
        ) = _sample_diversity(sa_res.counts, low_energy_states)
        row["total_seconds"] = time.perf_counter() - target_start
        return row
    except Exception as error:
        raise TargetEvaluationError(stage, error) from error


def _worker_initializer(args: argparse.Namespace) -> None:
    """Configure thread/GPU affinity and load one reusable scorer per process."""

    global _WORKER_ARGS, _WORKER_SCORER
    identity = mp.current_process()._identity  # stable ProcessPool worker ordinal
    worker_index = int(identity[-1] - 1) if identity else max(0, os.getpid() - 1)
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp_threads)
    if args.backend == "lightning.gpu":
        gpu_index = worker_index % args.gpu_count
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    torch.set_num_threads(args.omp_threads)
    torch.set_num_interop_threads(1)
    torch_device_name = args.torch_device
    if args.backend == "lightning.gpu" and torch_device_name == "auto":
        torch_device_name = "cuda:0"
    elif torch_device_name == "auto":
        torch_device_name = "cpu"
    torch_device = torch.device(torch_device_name)
    _WORKER_SCORER, _ = load_interface_scorer(
        args.checkpoint, torch_device=torch_device, seed=42 + worker_index
    )
    _WORKER_ARGS = args


def _evaluate_worker(graph_path_text: str) -> Dict[str, Any]:
    """Evaluate one target with complete exception containment inside a worker."""

    graph_path = Path(graph_path_text)
    started = time.perf_counter()
    try:
        if _WORKER_SCORER is None or _WORKER_ARGS is None:
            raise RuntimeError("Worker was not initialized")
        return evaluate_single_target(graph_path, _WORKER_SCORER, _WORKER_ARGS)
    except TargetEvaluationError as error:
        return _failure_row(graph_path, error, time.perf_counter() - started)
    except Exception as error:
        wrapped = TargetEvaluationError("worker_wrapper", error)
        return _failure_row(graph_path, wrapped, time.perf_counter() - started)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _failure_row(
    graph_path: Path,
    error: TargetEvaluationError,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Create a schema-complete failure row."""

    row: Dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "status": "failed",
            "target_id": graph_path.stem,
            "source_file": str(graph_path.resolve()),
            "total_seconds": elapsed_seconds,
            "error_stage": error.stage,
            "error_type": type(error.cause).__name__,
            "error_message": _clean_error_message(error.cause),
        }
    )
    return row


def _write_csv_atomic(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Atomically persist all rows so interrupted long runs remain recoverable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    os.replace(temporary, path)


def _append_csv_row(row: Mapping[str, Any], path: Path) -> None:
    """Append one completed target and force it from Python buffers to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _append_failure_log(
    path: Path, graph_path: Path, error: TargetEvaluationError
) -> None:
    """Append a durable, human-readable failure record without stopping the batch."""

    line = (
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}\t"
        f"{graph_path.resolve()}\t{error.stage}\t{type(error.cause).__name__}\t"
        f"{_clean_error_message(error.cause)}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _append_failure_row(path: Path, row: Mapping[str, Any]) -> None:
    """Persist a worker-returned failure row from the sole writer process."""

    line = (
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}\t"
        f"{row.get('source_file', '')}\t{row.get('error_stage', '')}\t"
        f"{row.get('error_type', '')}\t{row.get('error_message', '')}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _append_optimization_warning(path: Path, row: Mapping[str, Any]) -> None:
    """Persist non-fatal flat-landscape warnings from completed targets."""

    warning = str(row.get("optimization_warning", "")).strip()
    if not warning:
        return
    line = (
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}\t"
        f"{row.get('source_file', row.get('target_id', ''))}\t{warning}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_existing_csv(path: Path) -> List[Dict[str, Any]]:
    """Load prior rows for automatic interruption-safe resume."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric_values(rows: Iterable[Mapping[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key, "")
        if value not in ("", None):
            number = float(value)
            if math.isfinite(number):
                values.append(number)
    return values


def _mean_sd(rows: Sequence[Mapping[str, Any]], key: str, digits: int = 4) -> str:
    """Format mean ± sample SD, retaining a useful single-sample value."""

    values = _numeric_values(rows, key)
    if not values:
        return "N/A"
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    return f"{average:.{digits}f} ± {deviation:.{digits}f}"


def _median_iqr(rows: Sequence[Mapping[str, Any]], key: str, digits: int = 4) -> str:
    values = np.asarray(_numeric_values(rows, key), dtype=np.float64)
    if not len(values):
        return "N/A"
    q1, q3 = np.percentile(values, [25.0, 75.0])
    return f"{median(values):.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def _wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Return a Wilson 95% confidence interval for a binomial proportion."""

    if total <= 0:
        return math.nan, math.nan
    proportion = hits / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def _percent_summary(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values = _numeric_values(rows, key)
    if not values:
        return "N/A"
    deviation = stdev(values) if len(values) > 1 else 0.0
    return f"{100.0 * mean(values):.2f}% ± {100.0 * deviation:.2f}%"


def _hit_summary(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values = [int(round(value)) for value in _numeric_values(rows, key)]
    if not values:
        return "N/A"
    hits = sum(values)
    low, high = _wilson_interval(hits, len(values))
    return (
        f"{100.0 * hits / len(values):.2f}% ({hits}/{len(values)}; "
        f"95% CI {100.0 * low:.2f}–{100.0 * high:.2f}%)"
    )


def write_markdown_report(
    rows: Sequence[Mapping[str, Any]],
    report_path: Path,
    *,
    args: argparse.Namespace,
    available_count: int,
    selected_count: int,
    model_info: ModelLoadInfo,
    wall_seconds: float,
) -> None:
    """Write the paper-oriented aggregate benchmark report."""

    successful = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") != "success"]
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    checkpoint_note = (
        "已加载训练权重。"
        if model_info.status == "checkpoint_loaded"
        else "**警告：未使用训练权重；EGNN 剪枝来自固定种子的默认初始化，仅可用于工程自检。**"
    )

    lines = [
        "# SNAC 高柔性挑战集：XY-Mixer QAOA 与模拟退火基准",
        "",
        f"生成时间：{timestamp}",
        "",
        checkpoint_note,
        "",
        "## 运行配置",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| 数据目录 | `{Path(args.input_dir).resolve()}` |",
        f"| 可用图文件 | {available_count} |",
        f"| 本次选择目标 | {selected_count} |",
        f"| 成功 / 失败 | {len(successful)} / {len(failed)} |",
        f"| EGNN 权重状态 | `{model_info.status}` |",
        f"| EGNN 权重路径 | `{model_info.checkpoint_path.resolve()}` |",
        f"| 量子设备 | `{args.backend}` |",
        f"| 进程池 / 每进程 CPU 线程 | {args.effective_workers} / {args.omp_threads} |",
        f"| Python / PyTorch / PennyLane | `{sys.version.split()[0]}` / `{torch.__version__}` / `{qml.__version__}` |",
        f"| QAOA 深度 / 优化器 / 最大评估 | 2 / {args.qaoa_optimizer.upper()} / {args.qaoa_max_evals} |",
        f"| QAOA shots | {args.shots} |",
        f"| 模拟实现 | {args.simulation_mode}（经典模拟；subspace 为相同回路的合法子空间精确演化） |",
        f"| SA reads / sweeps | {args.sa_reads} / {args.sa_sweeps} |",
        f"| 总墙钟时间 | {wall_seconds:.2f} s |",
        "",
        "## 核心结果",
        "",
        "所有能量均来自不含独热罚项的同一物理目标。Hit Rate 以样本最优能量与精确基态能量在 `atol=1e-6, rtol=0` 下相等为准；95% CI 为 Wilson 区间。",
        "",
        "| 指标 | XY-Mixer QAOA | Simulated Annealing |",
        "|---|---:|---:|",
        f"| 有效样本数 | {len(successful)} | {len(successful)} |",
        f"| 平均比特数 | {_mean_sd(successful, 'num_bits', 2)} | {_mean_sd(successful, 'num_bits', 2)} |",
        f"| 平均合法构象空间 | {_mean_sd(successful, 'configuration_count', 2)} | {_mean_sd(successful, 'configuration_count', 2)} |",
        f"| 合法率 | {_percent_summary(successful, 'qaoa_legal_rate')} | {_percent_summary(successful, 'sa_legal_rate')} |",
        f"| 基态命中率 (Hit Rate) | {_hit_summary(successful, 'qaoa_hit_ground')} | {_hit_summary(successful, 'sa_hit_ground')} |",
        f"| 平均能隙残差 (Mean Gap) | {_mean_sd(successful, 'qaoa_gap', 6)} | {_mean_sd(successful, 'sa_gap', 6)} |",
        f"| 单次读出成功率 | {_percent_summary(successful, 'qaoa_success_probability')} | {_percent_summary(successful, 'sa_success_probability')} |",
        f"| 优化初态期望能量 | {_mean_sd(successful, 'optimization_energy_start', 6)} | — |",
        f"| 优化终态期望能量 | {_mean_sd(successful, 'optimization_energy_end', 6)} | — |",
        f"| 平均能量下降 | {_mean_sd(successful, 'optimization_energy_drop', 6)} | — |",
        f"| 平均收敛评估步数 | {_mean_sd(successful, 'optimization_steps_converged', 2)} | — |",
        f"| 构象熵（nats） | {_mean_sd(successful, 'qaoa_conformational_entropy', 4)} | {_mean_sd(successful, 'sa_conformational_entropy', 4)} |",
        f"| 低能态覆盖率 | {_percent_summary(successful, 'qaoa_low_energy_coverage')} | {_percent_summary(successful, 'sa_low_energy_coverage')} |",
        f"| 平均运行耗时 | {_mean_sd(successful, 'qaoa_seconds', 2)} s | {_mean_sd(successful, 'sa_seconds', 2)} s |",
        f"| 运行耗时中位数 [IQR] | {_median_iqr(successful, 'qaoa_seconds', 2)} s | {_median_iqr(successful, 'sa_seconds', 2)} s |",
        "",
        "## 评测队列特征",
        "",
        "| 指标 | 统计量（mean ± SD） |",
        "|---|---:|",
        f"| CDR-H3 长度 | {_mean_sd(successful, 'cdr3_len', 2)} aa |",
        f"| 原始复合物节点数 | {_mean_sd(successful, 'num_input_nodes', 2)} |",
        f"| 剪枝子图节点数 | {_mean_sd(successful, 'num_subgraph_nodes', 2)} |",
        f"| QUBO 位点数 | {_mean_sd(successful, 'num_selected_sites', 2)} |",
        f"| Frozen 背景残基数 | {_mean_sd(successful, 'num_frozen_residues', 2)} |",
        f"| 精确低能态数 | {_mean_sd(successful, 'low_energy_state_count', 2)} |",
        "",
        "## 流水线开销",
        "",
        "| 阶段 | 平均耗时（s，mean ± SD） |",
        "|---|---:|",
        f"| 图加载 | {_mean_sd(successful, 'load_seconds', 3)} |",
        f"| EGNN 前向与剪枝 | {_mean_sd(successful, 'pruning_seconds', 3)} |",
        f"| 力场与 QUBO 构建 | {_mean_sd(successful, 'qubo_seconds', 3)} |",
        f"| 精确合法空间枚举 | {_mean_sd(successful, 'ground_truth_seconds', 3)} |",
        f"| 单目标完整流水线 | {_mean_sd(successful, 'total_seconds', 2)} |",
        "",
        "## 指标定义",
        "",
        "- **合法率**：每个残基局部寄存器均满足 Hamming weight = 1 的样本占比。SA 只提出逐残基换态，因此理论及实测合法率均为 100%。",
        "- **能隙残差**：算法采得的最低物理能量减去精确枚举基态能量，越接近 0 越好。",
        "- **单次读出成功率**：QAOA 为 shots 中基态样本占比；SA 为独立 reads 中到达基态的占比。",
        "- **构象熵**：对实测离散构象频率计算 Shannon 熵 `-Σ p ln p`，单位为 nats。",
        "- **低能态覆盖率**：采到的不同低能构象数除以精确枚举中满足 `E ≤ E_ground + 2.0 kcal/mol` 的构象总数。",
            "- **平均耗时**：QAOA 包含经典参数优化和最终有限 shots 采样；SA 包含全部 reads 与 sweeps。",
            "- **优化初态/终态期望能量**：分别为 COBYLA 首次目标评估和最终参数的精确期望物理能量。",
            "- **优化步数**：优化器实际报告的目标函数评估次数（`nfev`）。",
        "",
        "## 异常清单",
        "",
    ]
    if not failed:
        lines.append("本次运行无失败或跳过条目。")
    else:
        lines.extend(
            [
                "| 目标 | 阶段 | 异常类型 | 信息 |",
                "|---|---|---|---|",
            ]
        )
        for row in failed:
            message = str(row.get("error_message", "")).replace("|", "\\|")
            lines.append(
                f"| `{row.get('target_id', '')}` | `{row.get('error_stage', '')}` | "
                f"`{row.get('error_type', '')}` | {message} |"
            )

    lines.extend(
        [
            "",
            "## 可复现性",
            "",
            f"- 明细 CSV：`{(report_path.parent / CSV_FILENAME).resolve()}`",
            "- 每个目标均使用量子/退火随机种子 42。",
            "- 启动时自动读取 CSV 并跳过已有目标；每个新结果立即 `flush()` 并同步到磁盘。",
            f"- 失败日志：`{(report_path.parent / FAILED_LOG_FILENAME).resolve()}`",
            f"- 收敛警示日志：`{(report_path.parent / OPTIMIZATION_WARNINGS_FILENAME).resolve()}`",
            "- QAOA 与 SA 均使用 `physical_self` 和 `physical_pair`，不使用 QUBO 独热罚项。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./dataset_clean/graphs/test_snac_hard"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./checkpoints/best_egnn_pruning.pt"),
    )
    parser.add_argument(
        "--backend",
        choices=("lightning.qubit", "lightning.gpu"),
        default="lightning.qubit",
        help="PennyLane simulator backend.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--simulation-mode", choices=("subspace", "pennylane"), default="subspace", help="Exact legal-subspace classical simulation, or full PennyLane statevector.")
    parser.add_argument("--optimization-timeout", type=float, default=1800, help="Seconds, checked between objective evaluations; cannot interrupt a native gate.")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--omp-threads", type=int, default=3)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument(
        "--gpu-workers-per-device",
        type=int,
        default=1,
        help="Concurrent statevector workers allowed per visible T4.",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument(
        "--qaoa-max-evals",
        type=int,
        default=90,
        help="Maximum COBYLA objective evaluations per target (default: 90).",
    )
    parser.add_argument(
        "--qaoa-optimizer",
        choices=("cobyla", "adam"),
        default="cobyla",
    )
    parser.add_argument("--sa-reads", type=int, default=50)
    parser.add_argument("--sa-sweeps", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("./benchmark_results"))
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated compatibility flag; resume is automatic.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore an existing CSV and start a new benchmark table.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_targets is not None and args.max_targets <= 0:
        raise ValueError("--max-targets must be positive")
    if args.optimization_timeout <= 0:
        raise ValueError("--optimization-timeout must be positive")
    if args.simulation_mode == "subspace" and args.qaoa_optimizer != "cobyla":
        raise ValueError("Subspace simulation requires COBYLA")
    if (
        args.shots <= 0
        or args.sa_reads <= 0
        or args.sa_sweeps <= 0
        or args.qaoa_max_evals <= 0
    ):
        raise ValueError(
            "--shots, --qaoa-max-evals, --sa-reads, and --sa-sweeps must be positive"
        )
    if args.workers <= 0 or args.omp_threads <= 0:
        raise ValueError("--workers and --omp-threads must be positive")
    if args.gpu_count <= 0 or args.gpu_workers_per_device <= 0:
        raise ValueError("GPU counts must be positive")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Challenge-set directory not found: {args.input_dir}")
    if args.backend == "lightning.gpu" and not torch.cuda.is_available():
        raise RuntimeError("lightning.gpu requires a CUDA-enabled PyTorch environment")
    if args.torch_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.torch_device}, but CUDA is unavailable")


def _run(args: argparse.Namespace) -> int:
    """Execute the fault-tolerant batch benchmark and write both deliverables."""

    if args.device is not None:
        if args.device not in {"lightning.qubit", "lightning.gpu"}:
            raise ValueError("--device must be lightning.qubit or lightning.gpu")
        args.backend = args.device
    _validate_args(args)
    all_graphs = sorted(args.input_dir.glob("*.pt"), key=lambda path: path.name.lower())
    if not all_graphs:
        raise FileNotFoundError(f"No .pt graphs found under {args.input_dir}")
    selected_graphs = (
        all_graphs[: args.max_targets]
        if args.max_targets is not None
        else all_graphs
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    config = {k: str(getattr(args, k)) for k in ('simulation_mode', 'backend', 'shots', 'qaoa_max_evals', 'qaoa_optimizer', 'sa_reads', 'sa_sweeps')}
    config['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint.exists() else 'missing'
    config['input_dir'] = str(args.input_dir.resolve())
    for script in ('qaoa_interface_sampler.py', 'batch_benchmark_hard_set.py', 'model_egnn_pruning.py', 'subgraph_to_qubo.py',
                   *SHARED_HELPER_MODULES):
        config[script] = hashlib.sha256(repo_path(script).read_bytes()).hexdigest()
    if not args.fresh and (args.out_dir / CSV_FILENAME).exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != config:
            raise ValueError('Existing results have different or unknown provenance; use a new --out-dir.')
    manifest_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    csv_path = args.out_dir / CSV_FILENAME
    report_path = args.out_dir / REPORT_FILENAME
    failed_log_path = args.out_dir / FAILED_LOG_FILENAME
    optimization_warnings_path = args.out_dir / OPTIMIZATION_WARNINGS_FILENAME
    selected_sources = {str(path.resolve()) for path in selected_graphs}
    rows = [] if args.fresh else [
        row
        for row in _read_existing_csv(csv_path)
        if str(Path(str(row.get("source_file", ""))).resolve()) in selected_sources and row.get("status") == "success"
    ]
    _write_csv_atomic(rows, csv_path)
    if args.fresh:
        failed_log_path.write_text("", encoding="utf-8")
        optimization_warnings_path.write_text("", encoding="utf-8")
    else:
        if not failed_log_path.exists():
            failed_log_path.touch()
        if not optimization_warnings_path.exists():
            optimization_warnings_path.touch()
    completed_files = {
        str(Path(str(row.get("source_file", ""))).resolve())
        for row in rows
        if row.get("source_file") and row.get("status") == "success"
    }
    pending = [
        path for path in selected_graphs if str(path.resolve()) not in completed_files
    ]
    resumed_target_seconds = sum(_numeric_values(rows, "total_seconds"))
    if args.backend == "lightning.gpu":
        args.effective_workers = min(
            args.workers, args.gpu_count * args.gpu_workers_per_device
        )
    else:
        args.effective_workers = args.workers
    args.effective_workers = max(
        1, min(args.effective_workers, max(1, len(selected_graphs)))
    )
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp_threads)

    scorer_probe, model_info = load_interface_scorer(
        args.checkpoint, torch_device=torch.device("cpu"), seed=42
    )
    del scorer_probe
    if model_info.warning:
        tqdm.write(f"WARNING: {model_info.warning}", file=sys.stderr)

    batch_start = time.perf_counter()
    success_count = sum(row.get("status") == "success" for row in rows)
    failure_count = len(rows) - success_count
    progress = tqdm(
        total=len(pending),
        desc="SNAC hard-set benchmark",
        unit="target",
        dynamic_ncols=True,
    )
    if pending:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.effective_workers,
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(args,),
        ) as executor:
            futures = {
                executor.submit(_evaluate_worker, str(path.resolve())): path
                for path in pending
            }
            for future in concurrent.futures.as_completed(futures):
                graph_path = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    wrapped = TargetEvaluationError("process_pool", error)
                    row = _failure_row(graph_path, wrapped, 0.0)
                rows.append(row)
                if row.get("status") == "success":
                    success_count += 1
                else:
                    failure_count += 1
                    _append_failure_row(failed_log_path, row)
                    tqdm.write(
                        f"FAILED {graph_path.name} at {row.get('error_stage', '')}: "
                        f"{row.get('error_type', '')}: {row.get('error_message', '')}",
                        file=sys.stderr,
                    )
                # Only the parent process writes. This is both lock-free and
                # immune to interleaved CSV records from concurrent workers.
                _append_csv_row(row, csv_path)
                _append_optimization_warning(optimization_warnings_path, row)
                progress.update(1)
                progress.set_postfix(
                    ok=success_count,
                    failed=failure_count,
                    pdb=row.get("pdb_id", graph_path.stem),
                )
    progress.close()

    wall_seconds = time.perf_counter() - batch_start
    write_markdown_report(
        rows,
        report_path,
        args=args,
        available_count=len(all_graphs),
        selected_count=len(selected_graphs),
        model_info=model_info,
        wall_seconds=wall_seconds,
    )
    final_success = sum(row.get("status") == "success" for row in rows)
    final_failed = len(rows) - final_success
    print(
        f"Benchmark complete: {final_success} successful, {final_failed} failed; "
        f"CSV={csv_path.resolve()}; report={report_path.resolve()}"
    )
    return 0
