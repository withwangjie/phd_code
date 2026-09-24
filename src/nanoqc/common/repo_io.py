"""Shared, dependency-free file helpers used across the pipeline.

Only helpers whose output is byte-for-byte identical at every former call site
live here. The several atomic JSON writers in this repository are deliberately
NOT merged into one: they write different formats (``allow_nan=False`` + fsync
vs. ``default=str`` + trailing newline, different indent/sort settings), and
those files are hashed into manifests, so unifying them would change results.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, os.PathLike]

# Repository layout. REPO_ROOT is the checkout root (it holds configs/, docs/,
# scripts/, .venv and .runtime/); every module lives under src/nanoqc/.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
DOCS_DIR = "docs"
CONFIGS_DIR = "configs"

# Bare module file name -> path relative to the repository root. Code
# fingerprints (code_sha256 dicts in run manifests) keep the bare names as keys
# so manifests stay comparable in structure; the file itself is found here.
MODULE_LAYOUT = {
    "repo_io.py": "src/nanoqc/common/repo_io.py",
    "seed_streams.py": "src/nanoqc/common/seed_streams.py",
    "prediction_contract.py": "src/nanoqc/common/prediction_contract.py",
    "residue_tables.py": "src/nanoqc/structure/residue_tables.py",
    "structural_quality.py": "src/nanoqc/structure/structural_quality.py",
    "evaluate_complex_metrics.py": "src/nanoqc/structure/evaluate_complex_metrics.py",
    "audit_all_datasets.py": "src/nanoqc/data/audit_all_datasets.py",
    "build_final_pyg_dataset.py": "src/nanoqc/data/build_final_pyg_dataset.py",
    "build_independence_cluster_map.py": "src/nanoqc/data/build_independence_cluster_map.py",
    "audit_external_vhh_independence.py": "src/nanoqc/data/audit_external_vhh_independence.py",
    "sequence_identity.py": "src/nanoqc/data/sequence_identity.py",
    "safe_graph_load.py": "src/nanoqc/data/safe_graph_load.py",
    "model_egnn_pruning.py": "src/nanoqc/model/model_egnn_pruning.py",
    "train_egnn_pruning.py": "src/nanoqc/model/train_egnn_pruning.py",
    "egnn_seed_sensitivity.py": "src/nanoqc/model/egnn_seed_sensitivity.py",
    "subgraph_to_qubo.py": "src/nanoqc/qubo/subgraph_to_qubo.py",
    "qaoa_interface_sampler.py": "src/nanoqc/solvers/qaoa_interface_sampler.py",
    "instance.py": "src/nanoqc/quantum/instance.py",
    "resource_estimation.py": "src/nanoqc/quantum/resource_estimation.py",
    "paired_statistics.py": "src/nanoqc/inference/paired_statistics.py",
    "analyze_quantum_scaling.py": "src/nanoqc/inference/analyze_quantum_scaling.py",
    "analyze_structure_recovery.py": "src/nanoqc/inference/analyze_structure_recovery.py",
    "analyze_quantum_exploration.py": "src/nanoqc/inference/analyze_quantum_exploration.py",
    "batch_benchmark_hard_set.py": "src/nanoqc/experiments/batch_benchmark_hard_set.py",
    "run_real_complex_pilot.py": "src/nanoqc/experiments/run_real_complex_pilot.py",
    "generate_energy_calibration_dataset.py": "src/nanoqc/experiments/generate_energy_calibration_dataset.py",
    "fit_qaoa_transfer_parameters.py": "src/nanoqc/experiments/fit_qaoa_transfer_parameters.py",
    "run_external_structure_baselines.py": "src/nanoqc/experiments/run_external_structure_baselines.py",
    "generate_final_research_report.py": "src/nanoqc/reporting/generate_final_research_report.py",
    "generate_figure1_pymol_script.py": "src/nanoqc/reporting/generate_figure1_pymol_script.py",
    "run_full_experiment.py": "src/nanoqc/pipeline/run_full_experiment.py",
    "resolve_server_config.py": "src/nanoqc/pipeline/resolve_server_config.py",
}


def repo_path(name: str, repo_root: PathLike | None = None) -> Path:
    """Absolute path of a project module given its bare file name (e.g. ``subgraph_to_qubo.py``)."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        return root / MODULE_LAYOUT[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a module of this repository") from None


def module_name(name: str) -> str:
    """Dotted module path for ``python -m`` given a bare file name."""
    return MODULE_LAYOUT[name][len("src/"):-len(".py")].replace("/", ".")


def sha256_file(path: PathLike) -> str:
    """Hex SHA-256 of a file's bytes, streamed (never loads the file whole)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json_fsync(path: Path, value: Any) -> None:
    """Commit one complete JSON record atomically (indent=2, NaN rejected, fsync).

    Exact format of the former ``batch_benchmark_hard_set._ablation_atomic_json``.
    """
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
