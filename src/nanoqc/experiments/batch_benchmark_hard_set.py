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

This module is the command-line entry point. ``main`` dispatches on the mode
flag; each mode lives in its own module in ``nanoqc.experiments``:

    (no flag)                 hard_set_evaluation    per-target benchmark + report
    --research-ablation       research_ablation      matched QAOA-vs-classical sweep
                              calibration_fit        (--fit-energy-calibration-csv)
    --paired-statistics       benchmark_statistics   paired cluster-level inference
    --structure-evaluation    structure_benchmarks   all-atom modes
    --allatom-experiment      structure_benchmarks
    --recovery-benchmark      structure_benchmarks
    --real-complex-pilot      run_real_complex_pilot

``SHARED_HELPER_MODULES`` (benchmark_common) lists the modules every mode
fingerprints in its provenance. Every name defined by those modules is still
importable from here, except the per-process worker state (``_WORKER_*``,
``_ABLATION_*``): it is set inside worker processes and a re-export would only
be a stale snapshot.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from filelock import FileLock

# Re-exported: historical import location of these model helpers.
from nanoqc.model.model_egnn_pruning import (  # noqa: F401
    ModelLoadInfo, _clean_error_message, _extract_state_dict, _normalize_state_dict_keys,
    _graph_protocol_signature, assert_checkpoint_graph_compatible, load_interface_scorer,
)
from nanoqc.inference.paired_statistics import paired_effect as _paired_effect  # noqa: F401

# Definitions now live in focused modules; re-exported so every existing
# `from nanoqc.experiments.batch_benchmark_hard_set import ...` keeps working.
from nanoqc.experiments.benchmark_common import (  # noqa: F401
    SHARED_HELPER_MODULES,
)
from nanoqc.experiments.hard_set_evaluation import (  # noqa: F401
    CSV_FILENAME,
    REPORT_FILENAME,
    FAILED_LOG_FILENAME,
    OPTIMIZATION_WARNINGS_FILENAME,
    CSV_FIELDS,
    TargetEvaluationError,
    _scalar,
    _zero_small_gap,
    _sample_diversity,
    evaluate_single_target,
    _worker_initializer,
    _evaluate_worker,
    _failure_row,
    _write_csv_atomic,
    _append_csv_row,
    _append_failure_log,
    _append_failure_row,
    _append_optimization_warning,
    _read_existing_csv,
    _numeric_values,
    _mean_sd,
    _median_iqr,
    _wilson_interval,
    _percent_summary,
    _hit_summary,
    write_markdown_report,
    _build_parser,
    _validate_args,
    _run,
)
from nanoqc.experiments.research_ablation import (  # noqa: F401
    _ablation_classical_counts,
    QTS_TARGET_CONFIDENCE,
    queries_to_solution,
    load_transfer_parameters,
    transfer_key,
    _exact_ground_metrics,
    _ablation_summarize,
    _ablation_run_case,
    _ablation_export_results,
    _ablation_append_results,
    _ablation_worker,
    _ablation_dispatch,
    _ablation_main,
    _time_budget_counts,
)
from nanoqc.experiments.calibration_fit import (  # noqa: F401
    fit_energy_calibration_csv,
)
from nanoqc.experiments.benchmark_statistics import (  # noqa: F401
    _paired_statistics_main,
)
from nanoqc.experiments.structure_benchmarks import (  # noqa: F401
    _structure_evaluation_main,
    _allatom_experiment_main,
    _recovery_comparison,
    _recovery_benchmark_main,
)



def main(argv: Optional[Sequence[str]] = None) -> int:
    """Hold an OS-backed lock throughout the run, including spawned workers."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--real-complex-pilot" in arguments:
        arguments.remove("--real-complex-pilot")
        from nanoqc.experiments.run_real_complex_pilot import main as real_complex_main
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


if __name__ == "__main__":
    raise SystemExit(main())
