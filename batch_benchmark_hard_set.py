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
import traceback
import warnings
import json
import hashlib
from filelock import FileLock
from seed_streams import derive_streams, derive_child_seed, save_stream_map, DEFAULT_MASTER_SEED
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
from tqdm.auto import tqdm

from model_egnn_pruning import EGNNInterfaceScorer, extract_top_interface_subgraph
from qaoa_interface_sampler import XYMixerQAOASampler
from subgraph_to_qubo import InterfaceQUBOBuilder


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


@dataclass(frozen=True)
class ModelLoadInfo:
    """Provenance of the EGNN weights used for pruning."""

    status: str
    checkpoint_path: Path
    warning: str = ""


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


def _clean_error_message(error: BaseException, limit: int = 1000) -> str:
    """Flatten an exception message for robust one-row CSV storage."""

    message = " ".join(str(error).replace("\x00", "").split())
    return message[:limit]


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    """Extract a model state dictionary from common checkpoint layouts."""

    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a state-dict mapping")
    for key in ("model_state_dict", "state_dict", "model", "scorer_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, torch.nn.Module):
            return candidate.state_dict()
        if isinstance(candidate, Mapping) and candidate:
            return candidate  # type: ignore[return-value]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("No model state dictionary was found in the checkpoint")


def _normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Strip wrapper prefixes added by DataParallel or training containers."""

    prefixes = ("module.", "model.", "scorer.")
    normalized: Dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = str(original_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        normalized[key] = value
    return normalized


def load_interface_scorer(
    checkpoint_path: Path,
    *,
    torch_device: torch.device,
    seed: int,
) -> Tuple[EGNNInterfaceScorer, ModelLoadInfo]:
    """Load trained EGNN weights or deterministically fall back to defaults."""

    torch.manual_seed(seed)
    scorer = EGNNInterfaceScorer()
    if not checkpoint_path.exists():
        warning = (
            f"EGNN checkpoint not found at {checkpoint_path}; using deterministic "
            "default initialization. Pruning scores are untrained."
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        scorer.to(torch_device).eval()
        return scorer, ModelLoadInfo("default_initialization", checkpoint_path, warning)

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config: Dict[str, Any] = {}
        if isinstance(payload, Mapping) and isinstance(payload.get("model_config"), Mapping):
            allowed = {
                "input_dim",
                "hidden_dim",
                "num_layers",
                "edge_attr_dim",
                "dropout",
                "coord_scale",
            }
            model_config = {
                str(key): value
                for key, value in payload["model_config"].items()
                if key in allowed
            }
        scorer = EGNNInterfaceScorer(**model_config)
        state_dict = _normalize_state_dict_keys(_extract_state_dict(payload))
        scorer.load_state_dict(state_dict, strict=True)
        scorer.to(torch_device).eval()
        return scorer, ModelLoadInfo("checkpoint_loaded", checkpoint_path)
    except Exception as error:  # checkpoint incompatibility must not stop the batch
        warning = (
            f"Could not load EGNN checkpoint {checkpoint_path}: "
            f"{type(error).__name__}: {_clean_error_message(error)}. "
            "Using deterministic default initialization instead."
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        torch.manual_seed(seed)
        scorer = EGNNInterfaceScorer().to(torch_device).eval()
        return scorer, ModelLoadInfo("checkpoint_load_failed", checkpoint_path, warning)


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
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
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
            max_active=15,
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
            max_sites=15,
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
    for script in ('qaoa_interface_sampler.py', 'batch_benchmark_hard_set.py', 'model_egnn_pruning.py', 'subgraph_to_qubo.py'):
        config[script] = hashlib.sha256(Path(__file__).with_name(script).read_bytes()).hexdigest()
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Hold an OS-backed lock throughout the run, including spawned workers."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--real-complex-pilot" in arguments:
        arguments.remove("--real-complex-pilot")
        from run_real_complex_pilot import main as real_complex_main
        return real_complex_main(arguments)
    if "--recovery-benchmark" in arguments:
        arguments.remove("--recovery-benchmark")
        return _recovery_benchmark_main(arguments)
    if "--allatom-experiment" in arguments:
        arguments.remove("--allatom-experiment")
        return _allatom_experiment_main(arguments)
    if "--structure-evaluation" in arguments:
        arguments.remove("--structure-evaluation")
        return _structure_evaluation_main(arguments)
    if "--paired-statistics" in arguments:
        arguments.remove("--paired-statistics")
        return _paired_statistics_main(arguments)
    if "--research-ablation" in arguments:
        arguments.remove("--research-ablation")
        return _ablation_main(arguments)
    args = _build_parser().parse_args(arguments)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.out_dir / '.benchmark.lock'), timeout=0):
        return _run(args)




# Research-mode orchestration lives in this original driver.
import itertools
import platform
from collections import Counter
from model_egnn_pruning import select_ablation_active, build_ablation_subgraph
import hashlib

def _ablation_digest(path: Path) -> str:
    """Streaming SHA256 for provenance."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def _ablation_atomic_json(path: Path, value: Any) -> None:
    """Commit one complete JSON record atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _ablation_classical_counts(sampler: Any, reads: int, seed: int,
                     greedy: bool, max_passes: int) -> tuple[dict, int]:
    """Uniform feasible sampling or multi-start coordinate descent."""
    rng = np.random.default_rng(seed)
    groups = list(sampler.site_to_variables.values())
    counts: Counter = Counter()
    evaluations = 0
    for _ in range(reads):
        chosen = [int(rng.choice(g)) for g in groups]
        bits = np.zeros(sampler.num_variables, dtype=np.int8)
        bits[chosen] = 1
        if greedy:
            current = sampler.physical_energy(bits)
            evaluations += 1
            for _ in range(max_passes):
                changed = False
                for site in rng.permutation(len(groups)):
                    old = chosen[site]
                    best, best_e = old, current
                    for proposal in groups[site]:
                        if proposal == old:
                            continue
                        trial = bits.copy()
                        trial[old], trial[proposal] = 0, 1
                        value = sampler.physical_energy(trial)
                        evaluations += 1
                        if value < best_e - 1e-12:
                            best, best_e = proposal, value
                    if best != old:
                        bits[old], bits[best] = 0, 1
                        chosen[site], current = best, best_e
                        changed = True
                if not changed:
                    break
        counts[tuple(map(int, bits))] += 1
    return dict(counts), evaluations


def _ablation_summarize(counts: dict, energies: dict, ground: float, window: float) -> dict:
    """Matched-output metrics; report low-energy mass separately from entropy."""
    n = sum(counts.values())
    low = {s for s, e in energies.items() if e <= ground + window}
    seen_low = low.intersection(counts)
    mass = sum(counts[s] for s in seen_low)
    probs = np.array(list(counts.values()), float) / n
    conditional = np.array([counts[s] / mass for s in seen_low]) if mass else np.array([])
    best = min(energies[s] for s in counts)
    return dict(outputs=n, best_energy=best, gap=max(0., best-ground),
                hit=int(abs(best-ground) <= 1e-6),
                ground_probability=sum(c for s,c in counts.items() if abs(energies[s]-ground)<=1e-6)/n,
                legal_rate=1.0, entropy=float(-sum(probs*np.log(probs))),
                low_energy_mass=mass/n, low_energy_coverage=len(seen_low)/len(low),
                low_energy_conditional_entropy=float(-sum(conditional*np.log(conditional))) if mass else None,
                unique_states=len(counts))


def _ablation_run_case(data: Any, scorer: Any, config: dict, args: Any, artifact: Path) -> list[dict]:
    """One paired instance: same physical objective, candidate set and QUBO for every solver AND every
    outputs/objective/restart combination swept below -- only the output-budget curve and QAOA's own
    objective/restart ablation vary within this one case; pruning/radius/depth/max_evals/seed (the
    shared input and fixed evaluation region) are fixed by ``config`` before any solver runs."""
    begin = time.perf_counter()
    active = select_ablation_active(data, config["pruning"], args.active_sites, config["seed"], scorer)
    sub = build_ablation_subgraph(data, active, config["radius"])
    qubo = InterfaceQUBOBuilder(min_variables=2*args.active_sites,
                              max_variables=2*args.active_sites, max_sites=args.active_sites).build(sub)
    # Independent optimize/sample seeds (never the shared perturb/input seed
    # config["seed"] above, and never each other): derived per-case, before
    # this function is ever called, from the master-seed optimize/sample
    # streams keyed by stable case labels (target/pruning/radius/depth/
    # max_evals/seed) -- see _ablation_main. Falls back to config["seed"]
    # only for a config dict built by older/external code that never set
    # these keys, so this stays runnable standalone.
    optimize_seed = config.get("optimize_seed", config["seed"])
    measurement_seed = config.get("measurement_seed", config["seed"])
    sample_seed = config.get("sample_seed", config["seed"])
    # ONE sampler instance: XYMixerQAOASampler.optimize_robust/.sample/
    # .simulated_annealing already accept explicit optimize_seed/
    # measurement_seed/sample_seed overrides directly, so a second instance
    # is not needed -- the base constructor seed below is only the fallback
    # used if a call site ever omits an explicit override.
    sampler = XYMixerQAOASampler(qubo.physical_self, qubo.physical_pair,
        qubo.site_to_variables, p=config["depth"], seed=optimize_seed, shots=args.outputs[0],
        simulation_mode="subspace", device_name="default.qubit")
    built = time.perf_counter()
    truth = sampler.enumerate_ground_states()
    energies = sampler.feasible_energy_map()
    oracle_seconds = time.perf_counter()-built
    low_ids = {s for s,e in energies.items() if e <= truth.energy+args.energy_window}
    assert low_ids
    records, raw = [], {}
    last_optimization = {}
    optimizations = {}
    # Optimize each QAOA (objective, restart-count) variant exactly once per
    # physical case. Output-budget curves differ only in final sampling shots;
    # repeating the same deterministic finite-shot optimization for every
    # outputs value wastes compute and injects wall-time jitter without
    # changing the optimized parameters.
    qaoa_cache = {}
    for outputs in args.outputs:
        for solver in ("qaoa", "sa", "uniform", "greedy"):
            if solver == "qaoa":
                for qaoa_objective, qaoa_restarts in itertools.product(args.qaoa_objective, args.qaoa_restarts):
                    cache_key = (qaoa_objective, qaoa_restarts)
                    if cache_key not in qaoa_cache:
                        optimize_start = time.perf_counter()
                        opt = sampler.optimize_robust(
                            max_evals=config["max_evals"], restarts=qaoa_restarts,
                            objective=qaoa_objective, cvar_alpha=args.cvar_alpha,
                            parameter_scale=args.parameter_scale, eval_shots=args.eval_shots,
                            optimize_seed=optimize_seed, measurement_seed=measurement_seed)
                        optimization_seconds = time.perf_counter() - optimize_start
                        optimization = dict(
                            success=opt.success, message=getattr(opt, "message", None),
                            evaluations=opt.evaluations, history=list(opt.history),
                            gammas=opt.gammas.tolist(), betas=opt.betas.tolist(),
                            termination_reason=getattr(opt, "termination_reason", None))
                        qaoa_cache[cache_key] = (opt, optimization, optimization_seconds)
                    opt, optimization, optimization_seconds = qaoa_cache[cache_key]
                    sample_start = time.perf_counter()
                    sampled = sampler.sample(
                        opt, shots=outputs, ground_state=truth, sample_seed=sample_seed)
                    sampling_seconds = time.perf_counter() - sample_start
                    counts = dict(sampled.counts)
                    last_optimization = optimization
                    # Logical solver budget remains optimization + this output
                    # sampling cost, even though optimization is physically
                    # reused across the output curve.
                    elapsed = optimization_seconds + sampling_seconds
                    if sum(counts.values()) != outputs or any(s not in energies for s in counts):
                        raise AssertionError("Sample count or local one-hot constraint violated.")
                    metrics = _ablation_summarize(counts, energies, truth.energy, args.energy_window)
                    # metrics owns outputs: it is the verified measured count.
                    records.append(dict(config, solver=solver,
                        qaoa_objective=qaoa_objective, qaoa_restarts=qaoa_restarts, eval_shots=args.eval_shots,
                        **metrics, num_bits=len(qubo.physical_self),
                        configuration_count=truth.configuration_count, solver_seconds=elapsed,
                        optimization_seconds=optimization_seconds,
                        sampling_seconds=sampling_seconds,
                        optimization_reused=(outputs != args.outputs[0]),
                        build_seconds=built-begin, oracle_seconds=oracle_seconds,
                        single_state_energy_queries=None,
                        optimizer_success=optimization.get("success"),
                        optimizer_evaluations=optimization.get("evaluations"),
                        termination_reason=optimization.get("termination_reason"),
                        optimization_energy_start=optimization["history"][0] if optimization.get("history") else None,
                        optimization_energy_end=opt.objective_value,
                        diagnostic_mean_energy_end=opt.energy,
                        total_opt_shots=opt.total_opt_shots))
                    records[-1].update(budget_mode="matched_outputs", budget_seconds=None, budget_overrun_seconds=0.0)
                    key = f"qaoa_obj{qaoa_objective}_restarts{qaoa_restarts}_outputs{outputs}"
                    optimizations[key] = dict(
                        optimization,
                        optimization_seconds=optimization_seconds,
                        reused_across_output_curve=True)
                    raw[key] = [{"bits":"".join(map(str,s)), "count":c} for s,c in sorted(counts.items())]
            else:
                start = time.perf_counter()
                queries = None
                if solver == "sa":
                    # Existing API calls each single-site proposal a sweep. Convert explicitly.
                    proposals = args.sa_passes * len(qubo.site_to_variables)
                    sampled = sampler.simulated_annealing(num_reads=outputs, site_passes=args.sa_passes, seed=sample_seed, ground_state=truth)
                    counts = dict(sampled.counts)
                    queries = outputs*(1+proposals)
                else:
                    counts, queries = _ablation_classical_counts(sampler, outputs, sample_seed+101,
                                                        solver=="greedy", args.greedy_passes)
                elapsed = time.perf_counter()-start
                if sum(counts.values()) != outputs or any(s not in energies for s in counts):
                    raise AssertionError("Sample count or local one-hot constraint violated.")
                metrics = _ablation_summarize(counts, energies, truth.energy, args.energy_window)
                # Do not pass outputs twice; _ablation_summarize supplies it.
                records.append(dict(config, solver=solver,
                    qaoa_objective=None, qaoa_restarts=None, eval_shots=None,
                    **metrics, num_bits=len(qubo.physical_self),
                    configuration_count=truth.configuration_count, solver_seconds=elapsed,
                    optimization_seconds=None, sampling_seconds=None,
                    optimization_reused=None,
                    build_seconds=built-begin, oracle_seconds=oracle_seconds,
                    single_state_energy_queries=queries,
                    optimizer_success=None, optimizer_evaluations=None, termination_reason=None,
                    optimization_energy_start=None, optimization_energy_end=None))
                records[-1].update(budget_mode="matched_outputs", budget_seconds=None, budget_overrun_seconds=0.0)
                raw[f"{solver}_outputs{outputs}"] = [{"bits":"".join(map(str,s)), "count":c} for s,c in sorted(counts.items())]
        if args.time_baselines:
            qaoa_rows_this_output = [
                r for r in records
                if r["solver"] == "qaoa"
                and r["outputs"] == outputs
                and r.get("qaoa_objective") == args.time_donor_objective
                and r.get("qaoa_restarts") == args.time_donor_restarts
            ]
            if len(qaoa_rows_this_output) != 1:
                raise ValueError(
                    f"Expected exactly one matched-time donor for outputs={outputs}, "
                    f"objective={args.time_donor_objective}, restarts={args.time_donor_restarts}; "
                    f"found {len(qaoa_rows_this_output)}")
            donor_row = qaoa_rows_this_output[0]
            if donor_row.get("termination_reason") == "all_restarts_failed":
                # No meaningful QAOA timing budget exists for a complete
                # optimization collapse; omit time baselines for this output.
                continue
            budget = donor_row["solver_seconds"]
            for method in ("sa", "uniform", "greedy"):
                counts, elapsed, queries = _time_budget_counts(
                    sampler, method, budget, sample_seed+10001,
                    args.sa_passes, args.greedy_passes)
                metrics = _ablation_summarize(counts, energies, truth.energy, args.energy_window)
                row = dict(donor_row)
                row.update(metrics, solver=method+"_time", reference_outputs=outputs, solver_seconds=elapsed,
                    single_state_energy_queries=queries, optimizer_success=None,
                    optimizer_evaluations=None, termination_reason=None,
                    optimization_energy_start=None, optimization_energy_end=None,
                    diagnostic_mean_energy_end=None, total_opt_shots=0,
                    budget_mode="matched_time_soft_deadline",
                    budget_seconds=budget, budget_overrun_seconds=max(0., elapsed-budget))
                records.append(row)
                raw[f"{method}_time_outputs{outputs}"] = [
                    {"bits":"".join(map(str,b)), "count":count}
                    for b,count in sorted(counts.items())]
    _ablation_atomic_json(artifact, dict(config=config, metrics=records, counts=raw,
        optimization=last_optimization, optimizations=optimizations,
        active_residue_ids=[data.residue_ids[i] for i in active.tolist()],
        frozen_residue_ids=[sub.residue_ids[i] for i in range(sub.num_nodes) if sub.is_frozen_environment[i]],
        physical_self=qubo.physical_self.tolist(), physical_pair=qubo.physical_pair.tolist(),
        site_to_variables=qubo.site_to_variables, variable_map=[vars(r) for r in qubo.variable_map],
        ground_energy=truth.energy, total_seconds=time.perf_counter()-begin,
        scope="coarse-grained fixed-backbone; classical exact subspace simulation; matched outputs only"))
    return records


def _ablation_export_results(out: Path) -> None:
    """Rebuild CSV from atomic per-case records after interruptions."""
    rows = []
    for path in sorted((out/"cases").glob("*.json")):
        rows.extend(json.loads(path.read_text())["metrics"])
    if not rows:
        return
    temp = out/"metrics.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, out/"metrics.csv")
    report = ["# Exploratory matched-output benchmark", "",
        "Classical simulation of a coarse-grained fixed-backbone model. No structural-accuracy or quantum-speedup claim.",
        "Rows are correlated target × seed × configuration runs, NOT independent proteins.",
        "Compare solvers within an instance. Do not compare raw energies across different prunings/radii.",
        "Exact oracle preprocessing is separately timed. Equal outputs do not mean equal compute budgets.",
        "", f"Completed cases: {len(list((out/'cases').glob('*.json')))}", "",
        "| Solver | Runs | Hit fraction | Mean gap | Low-energy coverage |",
        "|---|---:|---:|---:|---:|"]
    for solver in sorted({r["solver"] for r in rows}):
        group = [r for r in rows if r["solver"]==solver]
        report.append(f"| {solver} | {len(group)} | {np.mean([r['hit'] for r in group]):.4f} | "
                      f"{np.mean([r['gap'] for r in group]):.6g} | {np.mean([r['low_energy_coverage'] for r in group]):.4f} |")
    (out/"summary.md").write_text("\n".join(report), encoding="utf-8")


_ABLATION_SCORER = None


def _ablation_append_results(out: Path, key: str) -> None:
    """Parent-only durable append; atomic case files rebuild CSV after a crash."""
    rows = json.loads((out/"cases"/(key+".json")).read_text(encoding="utf-8"))["metrics"]
    path = out/"metrics.csv"
    new_file = not path.exists()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not new_file:
        with path.open(newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle))
    if any(set(row) - set(fields) for row in rows):
        raise ValueError("Case CSV schema changed within a frozen run")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _ablation_worker(task: tuple) -> tuple:
    """Own one atomic case file; leave aggregate CSV and failure logs to parent."""
    global _ABLATION_SCORER
    path, config, args, artifact, key = task
    try:
        torch.set_num_threads(args.omp_threads)
        if config["pruning"] == "egnn" and _ABLATION_SCORER is None:
            _ABLATION_SCORER, status = load_interface_scorer(
                args.checkpoint, torch_device=torch.device("cpu"), seed=42)
            if status.status != "checkpoint_loaded":
                raise ValueError("A trained matching checkpoint is mandatory")
            _ABLATION_SCORER.eval()
        data = torch.load(path, map_location="cpu", weights_only=False)
        config["pdb_id"] = getattr(data, "pdb_id", "")
        _ablation_run_case(data, _ABLATION_SCORER, config, args, artifact)
        return key, config, None
    except Exception:
        return key, config, traceback.format_exc()


def _ablation_dispatch(tasks: Iterable[tuple], workers: int) -> Iterable[tuple]:
    """Bound queued work; a crashed process propagates as a systemic failure."""
    if workers == 1:
        for task in tasks:
            yield _ablation_worker(task)
        return
    iterator = iter(tasks)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * workers:
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                else:
                    pending.add(pool.submit(_ablation_worker, task))
            if pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    yield future.result()


def _ablation_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Matched-output research ablations on the existing coarse-grained model.")
    parser.add_argument("--input-dir", type=Path, default=Path("dataset_clean_500/graphs/test_snac_hard"))
    parser.add_argument("--checkpoint", type=Path, default=Path("quantum-protein/checkpoints_500/best_egnn_pruning.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results_ablation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED,
        help="Derives independent, saved, per-case optimize/sample sub-seeds (see seed_streams.py) -- "
             "NEVER the same as --seeds, which are the shared perturb/site-selection repeat identities.")
    parser.add_argument("--pruning", nargs="+", choices=["egnn","contact","distance","cdr","random"], default=["egnn","contact","random"])
    parser.add_argument("--radii", type=float, nargs="+", default=[6.,10.])
    parser.add_argument("--depths", type=int, nargs="+", default=[1,2,3])
    parser.add_argument("--max-evals", type=int, nargs="+", default=[90,300])
    parser.add_argument("--active-sites", type=int, default=10)
    parser.add_argument("--outputs", type=int, nargs="+", default=[1000],
        help="Output-budget curve (e.g. 10 30 100 300 1000): each value is a fully matched-output "
             "comparison across all four solvers, so equal-output at one budget is never conflated "
             "with equal-output at a different budget, and neither is conflated with equal total "
             "compute -- see solver_seconds/oracle_seconds/single_state_energy_queries per row.")
    parser.add_argument("--qaoa-objective", nargs="+", choices=("mean","cvar"), default=["cvar"],
        help="QAOA optimizer objective ablation dimension (mean vs CVaR low-energy tail).")
    parser.add_argument("--qaoa-restarts", type=int, nargs="+", default=[4],
        help="QAOA multistart ablation dimension (1 = single-start, N = N-start COBYLA sharing one "
             "--max-evals budget per case, per optimize_robust's own documented budget sharing).")
    parser.add_argument("--cvar-alpha", type=float, default=.1)
    parser.add_argument("--eval-shots", type=int, default=500,
        help="Finite measurement shots per objective evaluation inside optimize_robust (never the "
             "exact analytic expectation) -- the NISQ-realistic finite-shot CVaR/mean estimate.")
    parser.add_argument("--parameter-scale", choices=("max_coefficient","feasible_iqr"), default="max_coefficient")
    parser.add_argument("--time-baselines", action="store_true", help="Add classical restart baselines using QAOA solver wall time; record soft-deadline overrun.")
    parser.add_argument("--time-donor-objective", choices=("mean","cvar"), default="cvar",
        help="QAOA objective whose wall time defines matched-time classical budgets.")
    parser.add_argument("--time-donor-restarts", type=int, default=4,
        help="QAOA restart count whose wall time defines matched-time classical budgets.")
    parser.add_argument("--sa-passes", type=int, default=100)
    parser.add_argument("--greedy-passes", type=int, default=50)
    parser.add_argument("--energy-window", type=float, default=2.)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--omp-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if (not 5 <= args.active_sites <= 15
            or min(*args.outputs, args.sa_passes, args.greedy_passes, *args.max_evals) <= 0
            or min(args.qaoa_restarts) <= 0 or args.eval_shots <= 0):
        parser.error("Require 5..15 sites and positive budgets (outputs/sa-passes/greedy-passes/max-evals/qaoa-restarts/eval-shots).")
    if any(not math.isfinite(r) or r<=0 for r in args.radii) or any(p not in (1,2,3) for p in args.depths):
        parser.error("Require positive finite radii and depths 1/2/3.")
    if args.max_targets < 0 or args.energy_window < 0 or not math.isfinite(args.energy_window):
        parser.error("Invalid target limit or energy window.")
    if args.time_baselines:
        if args.time_donor_objective not in args.qaoa_objective:
            parser.error("--time-donor-objective must be included in --qaoa-objective")
        if args.time_donor_restarts not in args.qaoa_restarts:
            parser.error("--time-donor-restarts must be included in --qaoa-restarts")
    if args.workers < 1 or args.omp_threads < 1:
        parser.error("workers and omp-threads must be positive")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.omp_threads)
    torch.set_num_threads(args.omp_threads)
    files = sorted(args.input_dir.glob("*.pt"))
    if args.max_targets:
        files = files[:args.max_targets]
    if not files:
        parser.error("No input graphs.")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out/".lock"), timeout=0):
        provenance = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            input_sha256={str(p.resolve()):_ablation_digest(p) for p in files},
            code_sha256={n:_ablation_digest(Path(__file__).parent/n) for n in
                ("model_egnn_pruning.py","subgraph_to_qubo.py","qaoa_interface_sampler.py","batch_benchmark_hard_set.py")},
            checkpoint_sha256=_ablation_digest(args.checkpoint) if "egnn" in args.pruning and args.checkpoint.exists() else None,
            python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
            pennylane=__import__("pennylane").__version__, scipy=__import__("scipy").__version__)
        manifest = out/"run_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError("Output provenance differs; use a new --out-dir.")
        _ablation_atomic_json(manifest, provenance)
        streams = derive_streams(args.master_seed)
        save_stream_map(out/"seed_streams.json", args.master_seed, streams)
        (out/"cases").mkdir(exist_ok=True)
        scorer = None
        if "egnn" in args.pruning:
            scorer, status = load_interface_scorer(args.checkpoint, torch_device=torch.device("cpu"), seed=42)
            if status.status != "checkpoint_loaded":
                raise ValueError("A trained matching checkpoint is mandatory; no random fallback.")
            scorer.eval()
        _ablation_export_results(out)
        settings = list(itertools.product(args.pruning,args.radii,args.depths,args.max_evals,args.seeds))
        total_planned = len(files)*len(settings)
        failed_keys_path = out/"failed_case_keys.json"
        failed_keys = set(json.loads(failed_keys_path.read_text())) if failed_keys_path.exists() else set()
        failed = 0
        tasks = []
        for path, setting in itertools.product(files, settings):
            config = dict(target=path.stem, pruning=setting[0], radius=setting[1],
                          depth=setting[2], max_evals=setting[3], seed=setting[4])
            labels = ("ablation", config["target"], config["pruning"], str(config["radius"]),
                      str(config["depth"]), str(config["max_evals"]), str(config["seed"]))
            for stream in ("optimize", "measurement", "sample"):
                config[stream+"_seed"] = derive_child_seed(streams[stream], *labels)
            key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
            artifact = out/"cases"/(key+".json")
            if artifact.exists():
                failed_keys.discard(key)
                continue
            tasks.append((path, config, args, artifact, key))
        for key, config, error in tqdm(_ablation_dispatch(tasks, args.workers),
                                      total=len(tasks), desc="Pending ablation cases"):
            if error is None:
                failed_keys.discard(key)
                _ablation_append_results(out, key)
            else:
                failed += 1
                failed_keys.add(key)
                with (out/"failed_cases.log").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(config)+"\n"+error+"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            _ablation_atomic_json(failed_keys_path, sorted(failed_keys))
        _ablation_atomic_json(failed_keys_path, sorted(failed_keys))
        _ablation_export_results(out)
        cases_completed_total = len(list((out/"cases").glob("*.json")))
        failures_total = len(failed_keys)
        gap = total_planned - cases_completed_total - failures_total
        # "Closed" per requirement #5: every planned case is accounted for
        # as EITHER a completed artifact OR an entry in the failure list --
        # never inferred from returncode or from any single file's mere
        # existence. A per-instance failure is expected and does not by
        # itself mean the stage failed; an unclosed gap (a case neither
        # completed nor recorded as failed -- e.g. the process was killed
        # mid-case) does.
        summary = dict(total_cases_planned=total_planned, cases_completed_total=cases_completed_total,
            failures_total=failures_total, gap=gap, closed=(gap==0),
            failures_this_invocation=failed)
        _ablation_atomic_json(out/"run_summary.json", summary)
        print(f"Results: {out}; failures this invocation: {failed}; "
              f"planned={total_planned} completed={cases_completed_total} failed_total={failures_total} closed={summary['closed']}")
        return 1 if (failed or not summary["closed"]) else 0



def _time_budget_counts(sampler: Any, method: str, seconds: float, seed: int,
                        sa_passes: int, greedy_passes: int) -> tuple[dict, float, int]:
    """Independent restarts to a soft wall deadline, checked between complete reads.

    Includes solver call/setup overhead. One final read may exceed the budget;
    this is recorded, never silently presented as strict equal-time computation.
    """
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("Time budget must be positive and finite")
    if method not in ("sa", "uniform", "greedy"):
        raise ValueError("Unknown baseline")
    start = time.perf_counter()
    counts: Counter = Counter()
    queries, restart = 0, 0
    while not counts or time.perf_counter()-start < seconds:
        if method == "sa":
            result = sampler.simulated_annealing(num_reads=1, site_passes=sa_passes,
                                                 seed=seed+restart)
            current = result.counts
            used = 1 + sa_passes*len(sampler.site_to_variables)
        else:
            current, used = _ablation_classical_counts(sampler, 1, seed+restart,
                                                       method == "greedy", greedy_passes)
            # Uniform draws need one objective query each for best-energy search.
            if method == "uniform":
                for bits in current:
                    sampler.physical_energy(bits)
                used = 1
        counts.update(current)
        queries += used
        restart += 1
    return dict(counts), time.perf_counter()-start, queries


def _paired_effect(values: Sequence[float], seed: int = 20260917,
                   resamples: int = 10000) -> dict:
    """Mean paired cluster difference, percentile CI, two-sided sign-flip test.

    Sign exchangeability/symmetry under the null and independent clusters are
    assumptions, not guaranteed by observational benchmark data.
    """
    d = np.asarray(values, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or resamples < 100:
        raise ValueError("Finite nonempty differences and >=100 resamples required")
    result = dict(n_clusters=len(d), mean_difference=float(d.mean()),
                  ci_low=None, ci_high=None, p_value=None)
    if len(d) < 2:
        return result
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, resamples, 256):
        size = min(256, resamples-start)
        boot.extend(d[rng.integers(0,len(d),(size,len(d)))].mean(axis=1))
    result.update(ci_low=float(np.quantile(boot,.025)), ci_high=float(np.quantile(boot,.975)))
    observed = abs(float(d.mean()))
    tolerance = 1e-12*max(1., observed)
    if len(d) <= 16:
        extreme = sum(abs(np.dot(signs,d)/len(d)) >= observed-tolerance
                      for signs in itertools.product((-1,1),repeat=len(d)))
        pvalue = extreme/(2**len(d))
    else:
        extreme = 0
        for start in range(0,resamples,256):
            size = min(256,resamples-start)
            signs = rng.choice([-1.,1.],size=(size,len(d)))
            extreme += int(np.count_nonzero(abs(signs@d/len(d)) >= observed-tolerance))
        pvalue = (extreme+1)/(resamples+1)
    result["p_value"] = float(pvalue)
    return result


def _holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error adjustment, original ordering retained."""
    p = np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid p values")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.
    for rank,index in enumerate(order):
        running = max(running,(len(p)-rank)*p[index])
        adjusted[index] = min(1.,running)
    return adjusted.tolist()


def _paired_statistics_main(argv: Optional[Sequence[str]] = None) -> int:
    """Analyse complete within-case pairs; average repeats within PDB/cluster."""
    parser = argparse.ArgumentParser(description="Exploratory paired cluster statistics")
    parser.add_argument("--results-dir",type=Path,required=True)
    parser.add_argument("--resamples",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=20260917)
    parser.add_argument("--cluster-map",type=Path,help="JSON mapping every PDB ID to antigen/sequence-family cluster")
    parser.add_argument("--budget-mode",choices=["outputs","time"],default="outputs")
    parser.add_argument("--primary-outputs", type=int, default=1000)
    parser.add_argument("--primary-objective", choices=("mean", "cvar"), default="cvar")
    parser.add_argument("--primary-restarts", type=int, default=4)
    parser.add_argument("--max-time-overrun-fraction",type=float,default=.10)
    args=parser.parse_args(argv)
    if args.resamples < 100 or not math.isfinite(args.max_time_overrun_fraction) or args.max_time_overrun_fraction<0:
        parser.error("Invalid resampling/overrun settings")
    paths=sorted((args.results_dir/"cases").glob("*.json"))
    if not paths:
        parser.error("No case JSON artifacts")
    cluster_map=json.loads(args.cluster_map.read_text()) if args.cluster_map else None
    grouped={}; skipped=Counter(); pair_count=Counter()
    metrics=("gap","hit","ground_probability","low_energy_mass","low_energy_coverage","entropy")
    for path in paths:
        case=json.loads(path.read_text())
        # Explicit primary output-budget contrast; never silently overwrite
        # all objective/restart/budget variants in a solver-keyed dictionary.
        selected = [r for r in case["metrics"] if
                    (r.get("reference_outputs", r.get("outputs")) if r["solver"].endswith("_time")
                     else r.get("outputs")) == args.primary_outputs]
        rows = {r["solver"]: r for r in selected if r["solver"] != "qaoa"}
        qrows = [r for r in selected if r["solver"] == "qaoa"]
        qrows = [r for r in qrows if r.get("qaoa_objective") == args.primary_objective
                 and r.get("qaoa_restarts") == args.primary_restarts]
        if len(qrows) != 1:
            skipped["missing_or_ambiguous_primary_contrast"] += 1
            continue
        rows["qaoa"] = qrows[0]
        if rows["qaoa"].get("termination_reason") == "all_restarts_failed":
            skipped["qaoa:all_restarts_failed"] += 1
            continue
        pdb=str(case["config"].get("pdb_id","")).strip().lower()
        if not pdb:
            raise ValueError(f"Missing PDB identity: {path}")
        if cluster_map is not None and pdb not in cluster_map:
            raise ValueError(f"Missing cluster mapping: {pdb}")
        cluster=str(cluster_map[pdb]) if cluster_map is not None else pdb
        for baseline in ("sa","uniform","greedy"):
            name=baseline if args.budget_mode=="outputs" else baseline+"_time"
            a,b=rows.get("qaoa"),rows.get(name)
            if a is None or b is None:
                skipped[name+":missing_pair"]+=1; continue
            if args.budget_mode=="outputs" and a["outputs"]!=b["outputs"]:
                skipped[name+":unequal_outputs"]+=1; continue
            if args.budget_mode=="time":
                budget=b.get("budget_seconds")
                if not budget or abs(budget-a["solver_seconds"])>1e-6*max(1.,budget):
                    skipped[name+":invalid_budget"]+=1; continue
                if b["budget_overrun_seconds"]>budget*args.max_time_overrun_fraction:
                    skipped[name+":overrun"]+=1; continue
            pair_count[name]+=1
            for metric in metrics:
                # Unequal read counts bias empirical diversity/coverage; do not test them in time mode.
                if args.budget_mode=="time" and metric not in ("gap","hit"):
                    continue
                av,bv=a.get(metric),b.get(metric)
                if av is None or bv is None or not np.isfinite([av,bv]).all():
                    skipped[name+":"+metric+":nonfinite"]+=1; continue
                grouped.setdefault((name,metric),{}).setdefault(cluster,{}).setdefault(pdb,[]).append(float(av)-float(bv))
    effects=[]; cluster_values=[]
    for (name,metric),clusters in sorted(grouped.items()):
        values=[]
        for cluster,pdbs in sorted(clusters.items()):
            # Equal weight to PDBs within families; repeated configs/seeds average within PDB.
            value=float(np.mean([np.mean(v) for v in pdbs.values()]))
            values.append(value)
            cluster_values.append(dict(baseline=name,metric=metric,cluster=cluster,difference=value,
                pdb_count=len(pdbs),paired_cases=sum(len(v) for v in pdbs.values())))
        effects.append(dict(baseline=name,metric=metric,**_paired_effect(values,args.seed,args.resamples)))
    tested=[e for e in effects if e["p_value"] is not None]
    for effect,pvalue in zip(tested,_holm_adjust([e["p_value"] for e in tested])):
        effect["p_holm"] = pvalue
    payload=dict(budget_mode=args.budget_mode,seed=args.seed,resamples=args.resamples,
        primary_outputs=args.primary_outputs, primary_objective=args.primary_objective,
        primary_restarts=args.primary_restarts,
        cluster_unit="provided family clusters" if cluster_map is not None else "PDB (homology dependence unresolved)",
        effects=effects,cluster_differences=cluster_values,paired_cases=dict(pair_count),exclusions=dict(skipped),
        source_sha256={p.name:_ablation_digest(p) for p in paths},
        cluster_map_sha256=_ablation_digest(args.cluster_map) if args.cluster_map else None,
        max_time_overrun_fraction=args.max_time_overrun_fraction,
        analysis_code_sha256=_ablation_digest(Path(__file__)))
    lines=["# Exploratory paired statistics", "", "Differences = QAOA - classical; negative gap favours QAOA, positive hit favours QAOA.",
        f"Primary contrast for both output- and time-budget analyses: outputs={args.primary_outputs}, objective={args.primary_objective}, restarts={args.primary_restarts}. Time baselines are generated from that same QAOA variant; complete all-restarts-failed collapses are excluded and counted explicitly.",
        "Repeats averaged within PDB, then PDBs within supplied families. Equal cluster weighting.",
        "95% percentile bootstrap intervals are marginal, not simultaneous. Two-sided sign-flip p values assume exchangeability/symmetry; Holm correction covers every tested contrast in this report.",
        "PDB clusters do not remove homologous-family dependence. Small cluster counts give unreliable intervals. One cluster: no CI or p value.",
        "Same-output budgets are not equal compute. Time mode compares best gap/hit only, excludes excessive soft-deadline overruns; unequal read entropy/coverage is not tested.",
        "Exact landscape preprocessing is excluded from solver time; these classical simulator runs do not establish hardware quantum advantage.",
        "", "| Baseline | Metric | Clusters | Mean difference | 95% CI | p | Holm p |", "|---|---|---:|---:|---|---|---|"]
    for e in effects:
        lines.append(f"| {e['baseline']} | {e['metric']} | {e['n_clusters']} | {e['mean_difference']:.6g} | {e['ci_low']}, {e['ci_high']} | {e['p_value']} | {e.get('p_holm')} |")
    lines += ["", "Pair/exclusion counts: "+json.dumps(dict(pairs=dict(pair_count),excluded=dict(skipped))),
        "", "Methods reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html"]
    out=args.results_dir/("statistics_"+args.budget_mode)
    with FileLock(str(out)+".lock",timeout=0):
        _ablation_atomic_json(out.with_suffix(".json"),payload)
        temp=out.with_suffix(".md.tmp")
        temp.write_text("\n".join(lines),encoding="utf-8")
        os.replace(temp,out.with_suffix(".md"))
    print(out.with_suffix(".md").resolve())
    return 0



def _structure_evaluation_main(argv: Optional[Sequence[str]] = None) -> int:
    """Evaluate explicit real-atom predictions; never reconstruct pseudo-atoms as native atoms.

    Manifest is a JSON list. Paths are relative to the manifest, not the CWD.
    Each case declares target, method, reference, prediction, active_residues,
    alignment_residues, partner_residues, selection_origin and protocol.
    """
    from subgraph_to_qubo import evaluate_atomistic_prediction
    parser = argparse.ArgumentParser(description="Real-atom structural evaluation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("structure_results"))
    args = parser.parse_args(argv)
    manifest = args.manifest.resolve()
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("Manifest must be a nonempty list")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # (Version Discrepancy remediation) evaluate_complex_metrics.py is now a
    # transitive dependency via evaluate_atomistic_prediction's call below,
    # so it is tracked here exactly like every other module this manifest
    # already fingerprints.
    source_hashes = {n:_ablation_digest(Path(__file__).parent/n) for n in
                     ("batch_benchmark_hard_set.py", "subgraph_to_qubo.py", "evaluate_complex_metrics.py")}
    import gemmi
    provenance = dict(manifest_sha256=_ablation_digest(manifest), code_sha256=source_hashes,
        numpy=np.__version__, gemmi=gemmi.__version__, python=sys.version)
    rows, failures = [], 0
    with FileLock(str(out/".lock"), timeout=0):
        record = out/"run_manifest.json"
        if record.exists() and json.loads(record.read_text()) != provenance:
            raise ValueError("Manifest/code/version changed; use a new output directory")
        _ablation_atomic_json(record, provenance)
        (out/"cases").mkdir(exist_ok=True)
        temp_csv=out/"structure_metrics.csv.tmp"
        with temp_csv.open("w",newline="",encoding="utf-8") as handle:
            writer=None
            for index, case in enumerate(tqdm(cases, desc="Real-atom evaluation")):
                try:
                    required=("target","method","reference","prediction","active_residues",
                              "alignment_residues","partner_residues","selection_origin","protocol")
                    if any(key not in case for key in required):
                        raise ValueError("Missing manifest fields: "+str(set(required)-case.keys()))
                    if case["protocol"] not in ("prediction","validation_control") or not case["selection_origin"]:
                        raise ValueError("Declare prediction/validation_control and selection_origin")
                    for key in ("active_residues","alignment_residues","partner_residues"):
                        if not isinstance(case[key],list) or any(not isinstance(r,str) for r in case[key]):
                            raise ValueError(f"{key} must be a list of author chain:residue IDs")
                    reference=(manifest.parent/str(case["reference"])).resolve()
                    prediction=(manifest.parent/str(case["prediction"])).resolve()
                    hashes=dict(reference=_ablation_digest(reference),prediction=_ablation_digest(prediction))
                    if hashes["reference"]==hashes["prediction"] and case["protocol"]!="validation_control":
                        raise ValueError("Identical reference/prediction must be labelled validation_control")
                    artifact=out/"cases"/f"{index:06d}.json"
                    signature=dict(case=case,structure_sha256=hashes)
                    if artifact.exists():
                        saved=json.loads(artifact.read_text())
                        if saved["signature"]!=signature:
                            raise ValueError("Input structure changed; use a new output directory")
                        result=saved["metrics"]
                    else:
                        result=evaluate_atomistic_prediction(reference,prediction,
                            active_residues=case["active_residues"], alignment_residues=case["alignment_residues"],
                            partner_residues=case["partner_residues"], model_index=int(case.get("model_index",0)),
                            contact_cutoff=float(case.get("contact_cutoff",5.)),
                            proximity_cutoff=float(case.get("proximity_cutoff",2.)),
                            chi1_tolerance=float(case.get("chi1_tolerance",20.)))
                        _ablation_atomic_json(artifact,dict(signature=signature,metrics=result))
                    row=dict(case_index=index,target=case["target"],method=case["method"],
                             protocol=case["protocol"],selection_origin=case["selection_origin"],**hashes)
                    row.update({k:v for k,v in result.items() if not isinstance(v,(list,dict))})
                    if writer is None:
                        writer=csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); handle.flush(); os.fsync(handle.fileno()); rows.append(row)
                except Exception:
                    failures+=1
                    with (out/"failed_cases.log").open("a",encoding="utf-8") as log:
                        log.write(json.dumps(dict(case_index=index,case=case))+"\n"+traceback.format_exc()+"\n")
                        log.flush(); os.fsync(log.fileno())
        os.replace(temp_csv,out/"structure_metrics.csv")
        report=["# Real-atom structural validation", "",f"Requested: {len(cases)}; successful: {len(rows)}; failed: {failures}.",
            "Validation controls are implementation checks, not structure prediction results.",
            "Alignment uses declared non-Active N/CA/C/O atoms. Active side chains are not independently fitted.",
            "Symmetry correction covers ASP/GLU/ARG/VAL/LEU/PHE/TYR; aromatic paired swaps are coupled.",
            "Chi1 is a partial torsion metric, not full rotamer recovery. Gly has no side-chain heavy atoms; Ala/Gly lack chi1.",
            "Contact precision/recall concern declared Active-partner residue pairs; partner selection defines the denominator.",
            "Severe proximity counts are geometric <2 A (unless configured) sidechain/partner pairs, not MolProbity clashscore or force-field energy.",
            "No all-atom candidate generation, minimization, docking or structural-accuracy claim is supplied by this evaluator.",
            "", "| Target | Method | Protocol | Side-chain RMSD (A) | Chi1 recovery | Contact F1 |", "|---|---|---|---:|---:|---:|"]
        for row in rows:
            report.append(f"| {row['target']} | {row['method']} | {row['protocol']} | {row['sidechain_rmsd_angstrom']} | {row['chi1_recovery_rate']} | {row['contact_f1']} |")
        report += ["", "Atom parsing follows Gemmi author IDs: https://gemmi.readthedocs.io/en/stable/mol.html"]
        temporary=out/"structure_report.md.tmp"
        temporary.write_text("\n".join(report),encoding="utf-8")
        os.replace(temporary,out/"structure_report.md")
    print(f"Structure results: {out}; failures: {failures}")
    return 1 if failures else 0



def _allatom_experiment_main(argv: Optional[Sequence[str]] = None) -> int:
    """One explicit all-atom case: QUBO -> matched-output solvers -> CIF -> evaluation."""
    from subgraph_to_qubo import AllAtomInterfaceQUBOBuilder, evaluate_atomistic_prediction
    import openmm
    parser=argparse.ArgumentParser(description="All-atom fixed-backbone chi1 experiment")
    parser.add_argument("--eval-shots",type=int,choices=(200,500,1000))
    parser.add_argument("--loop-relax-iterations",type=int,default=0)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--outputs",type=int,default=1000)
    parser.add_argument("--max-evals",type=int,default=90)
    parser.add_argument("--sa-passes",type=int,default=100)
    parser.add_argument("--robust-qaoa",action="store_true",help="Exact-subspace multistart optimization; total max-evals budget")
    parser.add_argument("--qaoa-restarts",type=int,default=4)
    parser.add_argument("--qaoa-objective",choices=("mean","cvar"),default="cvar")
    parser.add_argument("--cvar-alpha",type=float,default=.1)
    parser.add_argument("--parameter-scale",choices=("max_coefficient","feasible_iqr"),default="max_coefficient")
    parser.add_argument("--relax-iterations",type=int,default=200)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--optimize-seed",type=int,default=None,
        help="Independent optimizer sub-seed (defaults to --seed when omitted, for standalone-"
             "invocation compatibility). Drives XYMixerQAOASampler search RNG only.")
    parser.add_argument("--sample-seed",type=int,default=None,
        help="Independent final-output-sampling sub-seed (defaults to --seed when omitted). Drives "
             "the finite-shot draw for every solver reported output, never the QAOA search itself.")
    parser.add_argument("--measurement-seed",type=int,default=None,
        help="Independent in-search finite-shot measurement sub-seed (defaults to --seed when omitted). "
             "Drives optimize_robust's CVaR/mean objective-evaluation draws WHILE searching -- distinct "
             "from both --optimize-seed (restart initialization) and --sample-seed (final output draw).")
    args=parser.parse_args(argv)
    if min(args.outputs,args.max_evals,args.sa_passes)<=0 or args.relax_iterations<0:
        parser.error("Invalid budgets")
    optimize_seed = args.optimize_seed if args.optimize_seed is not None else args.seed
    measurement_seed = args.measurement_seed if args.measurement_seed is not None else args.seed
    sample_seed = args.sample_seed if args.sample_seed is not None else args.seed
    manifest=args.manifest.resolve(); case=json.loads(manifest.read_text())
    from prediction_contract import validate_prediction_contract
    validate_prediction_contract(case, manifest.parent)
    if case.get("protocol") not in ("validation_control","prediction") or not case.get("selection_origin"):
        parser.error("Manifest requires protocol and selection_origin")
    source=(manifest.parent/case["input_structure"]).resolve()
    reference=(manifest.parent/case["reference_structure"]).resolve() if case.get("reference_structure") else None
    if reference and _ablation_digest(source)==_ablation_digest(reference) and case["protocol"]!="validation_control":
        parser.error("Native-input controls must be declared validation_control")
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    provenance=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        resolved_optimize_seed=optimize_seed,resolved_measurement_seed=measurement_seed,resolved_sample_seed=sample_seed,
        manifest_sha256=_ablation_digest(manifest),input_sha256=_ablation_digest(source),
        reference_sha256=_ablation_digest(reference) if reference else None,
        code_sha256={n:_ablation_digest(Path(__file__).parent/n) for n in
            # (Version Discrepancy remediation) evaluate_complex_metrics.py
            # added: evaluate_atomistic_prediction (called via evaluate()
            # below) now depends on it as the single source of truth for
            # Fnat/iRMSD/LRMSD/DockQ/severe-clash metrics; structural_quality.py
            # is kept because evaluate_atomistic_prediction still sources the
            # legacy backbone-only DockQ variant from it (see subgraph_to_qubo.py).
            ("batch_benchmark_hard_set.py","subgraph_to_qubo.py","qaoa_interface_sampler.py",
             "prediction_contract.py","structural_quality.py","evaluate_complex_metrics.py")},openmm=openmm.__version__)
    with FileLock(str(out/".lock"),timeout=0):
        marker=out/"run_manifest.json"
        if marker.exists() and json.loads(marker.read_text())!=provenance:
            raise ValueError("All-atom provenance changed; use a new output directory")
        _ablation_atomic_json(marker,provenance)
        if (out/"completed.json").exists():
            saved=json.loads((out/"completed.json").read_text())
            if all((out/name).exists() and _ablation_digest(out/name)==digest for name,digest in saved["artifacts"].items()):
                print(f"Verified completed experiment: {out}");return 0
            raise ValueError("Completed artifacts changed or missing")
        print("Preparing complete atoms and Amber14 parameters",flush=True)
        builder=AllAtomInterfaceQUBOBuilder(source,case["active_residues"],seed=args.seed,
                                           chi1_angles=case.get("chi1_angles",[-60.,60.,180.]),
                                           candidate_relax_iterations=int(case.get("candidate_relax_iterations",0)))
        builder.write_structure(builder.base_positions,out/"prepared_input.cif")
        # Candidate coordinates make reconstruction independently auditable.
        np.savez_compressed(out/"candidate_coordinates.npz",base_positions_nm=builder.base_positions,
            **{f"indices_{i}":c["indices"] for i,c in enumerate(builder.candidates)},
            **{f"positions_nm_{i}":c["positions"] for i,c in enumerate(builder.candidates)})
        print("Decomposing full force-field energies and checking equivalence",flush=True)
        qubo=builder.build();qubo.export(out,"allatom")
        # ONE sampler instance: XYMixerQAOASampler.optimize_robust/.sample/
        # .simulated_annealing already accept explicit optimize_seed/
        # measurement_seed/sample_seed overrides directly.
        sampler=XYMixerQAOASampler(qubo.physical_self,qubo.physical_pair,qubo.site_to_variables,
            simulation_mode="subspace",p=2,seed=optimize_seed,shots=args.outputs)
        truth=sampler.enumerate_ground_states();energies=sampler.feasible_energy_map()
        records=[]
        def evaluate(path):
            return evaluate_atomistic_prediction(reference,path,active_residues=case["active_residues"],
                alignment_residues=case["alignment_residues"],partner_residues=case["partner_residues"])
        if reference:
            _ablation_atomic_json(out/"initial_structure_metrics.json",evaluate(out/"prepared_input.cif"))
        relax_only_path=out/"relax_only.cif"
        relax_only=builder.relax_positions(builder.base_positions,relax_only_path,
                                           minimize_iterations=args.relax_iterations)
        if args.loop_relax_iterations:
            relax_only.update(builder.relax_cdr_loop(relax_only_path,case.get('cdr3_residues',[]),iterations=args.loop_relax_iterations))
        _ablation_atomic_json(out/"relax_only_result.json",dict(relaxation=relax_only,
            structure_after_relaxation=evaluate(relax_only_path) if reference else None))
        with (out/"allatom_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=None
            for method in ("qaoa","sa","uniform","greedy"):
                print(f"Sampling and reconstructing {method}",flush=True)
                start=time.perf_counter()
                if method=="qaoa":
                    # (Version Discrepancy remediation) optimize_robust's
                    # CVaR-quantile keyword is cvar_alpha, not the bare alpha
                    # used in earlier revisions of this call site -- passing
                    # alpha= against the current qaoa_interface_sampler.py
                    # raises TypeError immediately (an unexpected-keyword
                    # error, not a silent misconfiguration), but the fix
                    # keeps this entry point in sync with the single source
                    # of truth rather than leaving it broken.
                    #
                    # --eval-shots has no CLI default (None), and
                    # --robust-qaoa can be passed without it -- this used to
                    # fall back to the now-removed analytic-expectation
                    # mode, which optimize_robust no longer accepts
                    # (eval_shots must be a positive integer). Resolve to
                    # this project's default of 500 whenever --robust-qaoa
                    # is set but --eval-shots wasn't, so this entry point
                    # never passes eval_shots=None into optimize_robust.
                    resolved_eval_shots=args.eval_shots or 500
                    opt=(sampler.optimize_robust(max_evals=args.max_evals,restarts=args.qaoa_restarts,
                        objective=args.qaoa_objective,cvar_alpha=args.cvar_alpha,parameter_scale=args.parameter_scale,eval_shots=resolved_eval_shots,
                        optimize_seed=optimize_seed,measurement_seed=measurement_seed)
                        if args.robust_qaoa or args.eval_shots else sampler.optimize(method="cobyla",max_iterations=args.max_evals))
                    sampled=sampler.sample(opt,shots=args.outputs,ground_state=truth,sample_seed=sample_seed);counts=dict(sampled.counts)
                    # Honest convergence/measurement-accounting fields
                    # (optimizer_success, termination_reason, gamma_best,
                    # beta_best, best_mean_energy) mirrored into the saved
                    # artifact alongside the pre-existing fields, so a saved
                    # optimization.json is fully self-describing without
                    # requiring the reader to re-derive them from history.
                    _ablation_atomic_json(out/"optimization.json",dict(energy=opt.energy,history=list(opt.history),
                        success=opt.success,evaluations=opt.evaluations,gammas=opt.gammas.tolist(),betas=opt.betas.tolist(),
                        objective_name=opt.objective_name,objective_value=opt.objective_value,
                        restart_records=opt.restart_records,history_kind=opt.objective_name,
                        total_opt_shots=opt.total_opt_shots,
                        eval_shots=resolved_eval_shots if (args.robust_qaoa or args.eval_shots) else None,
                        optimizer_success=opt.optimizer_success,termination_reason=opt.termination_reason,
                        gamma_best=opt.gamma_best.tolist() if opt.gamma_best is not None else None,
                        beta_best=opt.beta_best.tolist() if opt.beta_best is not None else None,
                        best_mean_energy=opt.best_mean_energy))
                elif method=="sa":
                    counts=dict(sampler.simulated_annealing(num_reads=args.outputs,site_passes=args.sa_passes,seed=sample_seed,ground_state=truth).counts)
                else:
                    counts,_=_ablation_classical_counts(sampler,args.outputs,sample_seed+101,method=="greedy",50)
                elapsed=time.perf_counter()-start
                selected=min(counts,key=lambda bits:(energies[bits],bits))
                prediction=out/(method+"_relaxed.cif")
                relaxation=builder.reconstruct(selected,prediction,minimize_iterations=args.relax_iterations)
                if args.loop_relax_iterations:
                    relaxation.update(builder.relax_cdr_loop(prediction,case.get('cdr3_residues',[]),iterations=args.loop_relax_iterations))
                if sum(counts.values()) != args.outputs:
                    raise AssertionError('Solver output budget mismatch')
                frequencies=np.array(list(counts.values()),dtype=float)/args.outputs
                budget_metrics=dict(total_opt_shots=opt.total_opt_shots if method=='qaoa' else 0,
                    bitstring_entropy=float(-np.sum(frequencies*np.log(frequencies))),
                    bitstring_entropy_upper_bound=float(np.log(args.outputs)),
                    low_energy_fraction=sum(c for b,c in counts.items() if energies[b]<=truth.energy+2.)/args.outputs)
                expected=energies[selected]+qubo.metadata["physical_constant_offset"]
                if not np.isclose(expected,relaxation["discrete_energy_kcal"],atol=1e-4,rtol=1e-9):
                    raise AssertionError("Selected structure and QUBO energy disagree")
                before=evaluate(prediction.with_name(prediction.stem+"_discrete.cif")) if reference else None
                after=evaluate(prediction) if reference else None
                result=dict(**budget_metrics,method=method,protocol=case["protocol"],selected_bits=list(selected),
                    counts=[dict(bits=list(b),count=c) for b,c in sorted(counts.items())],
                    sampling=_ablation_summarize(counts,energies,truth.energy,2.),relaxation=relaxation,
                    structure_before_relaxation=before,structure_after_relaxation=after,solver_seconds=elapsed)
                _ablation_atomic_json(out/(method+"_result.json"),result)
                quality_keys=('dockq_score','fnat','irmsd','lrmsd','dockq_category','num_severe_clashes','has_severe_clash','dockq_definition','dockq_backbone_score')
                result.update({k:after[k] for k in quality_keys} if after else {})
                _ablation_atomic_json(out/(method+"_result.json"),result)
                row=dict(**budget_metrics,**({k:after[k] for k in quality_keys} if after else {}),method=method,protocol=case["protocol"],outputs=sum(counts.values()),
                    bits=len(selected),ground_gap=energies[selected]-truth.energy,solver_seconds=elapsed,**relaxation,
                    sidechain_rmsd_before=before["sidechain_rmsd_angstrom"] if before else None,
                    sidechain_rmsd_after=after["sidechain_rmsd_angstrom"] if after else None)
                if writer is None:
                    writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);handle.flush();os.fsync(handle.fileno());records.append(row)
        lines=["# All-atom fixed-backbone experiment", "", "Protocol: "+case["protocol"],
            "Amber14 vacuum potential; not binding free energy. Input-conditioned distal chi, uniform chi1 grid; not a full rotamer library.",
            "Same output count and same relaxation conditions; not equal total computational cost. The native reference never enters candidate selection or energy ranking.",
            "Only the lowest discrete-energy sampled state per method is relaxed; no reference-based selection. Iteration cap does not guarantee convergence.",
            "Sampling uses classical exact-subspace simulation. These results do not establish quantum advantage.",
            "", "| Method | Discrete energy (kcal/mol) | Relaxed energy | SC RMSD before | SC RMSD after |", "|---|---:|---:|---:|---:|"]
        for r in records:
            lines.append(f"| {r['method']} | {r['discrete_energy_kcal']:.6g} | {r['relaxed_energy_kcal']:.6g} | {r['sidechain_rmsd_before']} | {r['sidechain_rmsd_after']} |")
        (out/"allatom_report.md").write_text("\n".join(lines),encoding="utf-8")
        files=[p for p in out.iterdir() if p.is_file() and p.name not in (".lock","completed.json")]
        _ablation_atomic_json(out/"completed.json",dict(artifacts={p.name:_ablation_digest(p) for p in files}))
        print(out/"allatom_report.md")
    return 0



def _recovery_comparison(initial: float, relax_only: float, final: float) -> dict:
    """Positive gains mean structural improvement; negative values are retained."""
    if not np.isfinite([initial,relax_only,final]).all():
        raise ValueError("Recovery RMSDs must be finite")
    return dict(initial_rmsd=initial,relax_only_rmsd=relax_only,final_rmsd=final,
        improvement_vs_input=initial-final,improvement_vs_relax_only=relax_only-final,
        better_than_input=bool(final<initial-1e-6),better_than_relax_only=bool(final<relax_only-1e-6))


def _recovery_benchmark_main(argv: Optional[Sequence[str]] = None) -> int:
    """Retrospective multi-seed chi1 perturbation/recovery, with relax-only control."""
    from subgraph_to_qubo import AllAtomInterfaceQUBOBuilder
    import openmm
    parser=argparse.ArgumentParser(description="Fixed-backbone perturbation-recovery control")
    parser.add_argument("--eval-shots",type=int,choices=(200,500,1000))
    parser.add_argument("--loop-relax-iterations",type=int,default=0)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--optimize-seeds",nargs="+",type=int,default=None,
        help="Per-seed independent optimizer sub-seeds, positionally matched to --seeds "
             "(defaults elementwise to --seeds when omitted, for standalone-invocation compatibility). "
             "Never the same values as --seeds or --sample-seeds.")
    parser.add_argument("--sample-seeds",nargs="+",type=int,default=None,
        help="Per-seed independent final-sampling sub-seeds, positionally matched to --seeds "
             "(defaults elementwise to --seeds when omitted).")
    parser.add_argument("--measurement-seeds",nargs="+",type=int,default=None,
        help="Per-seed independent in-search measurement sub-seeds, positionally matched to --seeds "
             "(defaults elementwise to --seeds when omitted). Drives optimize_robust's finite-shot "
             "CVaR/mean draws WHILE searching -- distinct from --optimize-seeds and --sample-seeds.")
    parser.add_argument("--min-perturb-degrees",type=float,default=40.)
    parser.add_argument("--max-perturb-degrees",type=float,default=120.)
    parser.add_argument("--outputs",type=int,default=1000)
    parser.add_argument("--max-evals",type=int,default=90)
    parser.add_argument("--sa-passes",type=int,default=100)
    parser.add_argument("--relax-iterations",type=int,default=200)
    parser.add_argument("--robust-qaoa",action="store_true")
    parser.add_argument("--qaoa-restarts",type=int,default=4)
    parser.add_argument("--qaoa-objective",choices=("mean","cvar"),default="cvar")
    parser.add_argument("--cvar-alpha",type=float,default=.1)
    parser.add_argument("--parameter-scale",choices=("max_coefficient","feasible_iqr"),default="max_coefficient")
    args=parser.parse_args(argv)
    if len(args.seeds)!=len(set(args.seeds)) or min(args.seeds)<0:
        parser.error("Use unique nonnegative seeds")
    if args.optimize_seeds is not None and len(args.optimize_seeds)!=len(args.seeds):
        parser.error("--optimize-seeds must match --seeds length")
    if args.sample_seeds is not None and len(args.sample_seeds)!=len(args.seeds):
        parser.error("--sample-seeds must match --seeds length")
    if args.measurement_seeds is not None and len(args.measurement_seeds)!=len(args.seeds):
        parser.error("--measurement-seeds must match --seeds length")
    if not 0<args.min_perturb_degrees<=args.max_perturb_degrees<=180:
        parser.error("Invalid perturbation angle range")
    if min(args.outputs,args.max_evals,args.sa_passes)<=0 or args.relax_iterations<0:
        parser.error("Invalid solver budgets")
    manifest=args.manifest.resolve();case=json.loads(manifest.read_text())
    native=(manifest.parent/case["native_structure"]).resolve()
    for key in ("active_residues","alignment_residues","partner_residues","selection_origin"):
        if not case.get(key): parser.error("Missing "+key)
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    provenance=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        input_sha256=_ablation_digest(native),manifest_sha256=_ablation_digest(manifest),
        code_sha256={n:_ablation_digest(Path(__file__).parent/n) for n in
            ("batch_benchmark_hard_set.py","subgraph_to_qubo.py","qaoa_interface_sampler.py")},openmm=openmm.__version__)
    rows=[];failures=0
    with FileLock(str(out/".lock"),timeout=0):
        record=out/"run_manifest.json"
        if record.exists() and json.loads(record.read_text())!=provenance:
            raise ValueError("Recovery provenance changed; use a new output directory")
        _ablation_atomic_json(record,provenance)
        generator=AllAtomInterfaceQUBOBuilder(native,case["active_residues"],seed=42)
        with (out/"recovery_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=None
            for idx, seed in enumerate(args.seeds):
                optimize_seed = args.optimize_seeds[idx] if args.optimize_seeds is not None else seed
                measurement_seed = args.measurement_seeds[idx] if args.measurement_seeds is not None else seed
                sample_seed = args.sample_seeds[idx] if args.sample_seeds is not None else seed
                try:
                    directory=out/f"seed_{seed}";directory.mkdir(exist_ok=True)
                    perturbed=directory/"perturbed_input.cif";metadata=directory/"perturbation.json"
                    if metadata.exists():
                        saved=json.loads(metadata.read_text())
                        if not perturbed.exists() or _ablation_digest(perturbed)!=saved["structure_sha256"]:
                            raise ValueError("Perturbed input changed or missing")
                    else:
                        positions,angles=generator.perturb_chi1(seed,args.min_perturb_degrees,args.max_perturb_degrees)
                        generator.write_structure(positions,perturbed)
                        _ablation_atomic_json(metadata,dict(seed=seed,angles=angles,structure_sha256=_ablation_digest(perturbed),
                            protocol="retrospective chi1-only recovery; no clash/energy/reference-based rejection"))
                    child_case=dict(input_structure=str(perturbed),reference_structure=str(native),
                        cdr3_residues=case.get('cdr3_residues',[]),pruning=case.get('pruning'),
                        candidate_relax_iterations=int(case.get("candidate_relax_iterations",0)),
                        active_residues=case["active_residues"],alignment_residues=case["alignment_residues"],
                        partner_residues=case["partner_residues"],protocol="validation_control",
                        selection_origin=case["selection_origin"]+"; retrospective fixed-backbone perturbation-recovery")
                    child_manifest=directory/"experiment.json"
                    _ablation_atomic_json(child_manifest,child_case)
                    experiment=directory/"experiment"
                    _allatom_experiment_main(["--manifest",str(child_manifest),"--out-dir",str(experiment),
                        "--outputs",str(args.outputs),"--max-evals",str(args.max_evals),"--sa-passes",str(args.sa_passes),
                        "--relax-iterations",str(args.relax_iterations),"--seed",str(seed),
                        "--optimize-seed",str(optimize_seed),"--measurement-seed",str(measurement_seed),"--sample-seed",str(sample_seed),
                        "--loop-relax-iterations",str(args.loop_relax_iterations),
                        *(["--eval-shots",str(args.eval_shots)] if args.eval_shots else []),
                        *(["--robust-qaoa","--qaoa-restarts",str(args.qaoa_restarts),
                           "--qaoa-objective",args.qaoa_objective,"--cvar-alpha",str(args.cvar_alpha),
                           "--parameter-scale",args.parameter_scale] if args.robust_qaoa else [])])
                    initial=json.loads((experiment/"initial_structure_metrics.json").read_text())
                    control=json.loads((experiment/"relax_only_result.json").read_text())
                    for method in ("qaoa","sa","uniform","greedy"):
                        result=json.loads((experiment/(method+"_result.json")).read_text())
                        final=result["structure_after_relaxation"]
                        recovery=_recovery_comparison(initial["sidechain_rmsd_angstrom"],
                            control["structure_after_relaxation"]["sidechain_rmsd_angstrom"],final["sidechain_rmsd_angstrom"])
                        final_energy=result["relaxation"].get("stage2_physical_energy_kcal",result["relaxation"]["relaxed_energy_kcal"])
                        energy_drop=result["relaxation"]["discrete_energy_kcal"]-final_energy
                        rmsd_before=result["structure_before_relaxation"]["sidechain_rmsd_angstrom"]
                        row=dict(target=case.get("target",native.stem),seed=seed,optimize_seed=optimize_seed,
                            measurement_seed=measurement_seed,sample_seed=sample_seed,method=method,**recovery,
                            **{k:result[k] for k in ('dockq_score','fnat','irmsd','lrmsd','dockq_category','num_severe_clashes','has_severe_clash','total_opt_shots','bitstring_entropy','low_energy_fraction','dockq_definition','dockq_backbone_score')},
                            chi1_recovery_initial=initial["chi1_recovery_rate"],chi1_recovery_final=final["chi1_recovery_rate"],
                            contact_f1_initial=initial["contact_f1"],contact_f1_final=final["contact_f1"],
                            relaxation_energy_drop=energy_drop,
                            relaxation_energy_down_rmsd_up=bool(energy_drop>1e-6 and final["sidechain_rmsd_angstrom"]>rmsd_before+1e-6))
                        if writer is None:
                            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);handle.flush();os.fsync(handle.fileno());rows.append(row)
                except Exception:
                    failures+=1
                    with (out/"failed_cases.log").open("a",encoding="utf-8") as f:
                        f.write(f"seed={seed}\n"+traceback.format_exc()+"\n");f.flush()
        lines=["# Retrospective chi1 perturbation-recovery", "",
            f"One target; requested seeds={len(args.seeds)}, failed seeds={failures}. Repeated seeds are not independent proteins; no inferential p values.",
            "Positive RMSD gains mean improvement. All methods share the perturbed input and relaxation protocol. Relax-only isolates local minimization without discrete search.",
            "Native backbone and distal chi remain fixed/inherited; this is neither de novo prediction nor an independent structural benchmark.",
            "Perturbations are not rejected based on energy or reference similarity. Failures must be included in the denominator; energy decrease alone is not accuracy.",
            "", "| Method | Successful seeds | Mean gain vs input (A) | Mean gain vs relax-only (A) | Energy-down/RMSD-up cases |", "|---|---:|---:|---:|---:|"]
        for method in ("qaoa","sa","uniform","greedy"):
            group=[r for r in rows if r["method"]==method]
            if group:
                lines.append(f"| {method} | {len(group)} | {np.mean([r['improvement_vs_input'] for r in group]):.6g} | {np.mean([r['improvement_vs_relax_only'] for r in group]):.6g} | {sum(r['relaxation_energy_down_rmsd_up'] for r in group)} |")
        (out/"recovery_report.md").write_text("\n".join(lines),encoding="utf-8")
        print(out/"recovery_report.md")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
