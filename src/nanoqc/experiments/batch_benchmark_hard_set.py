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
fingerprints in its provenance. Import a mode's functions from its own module;
this entry point imports only what ``main`` dispatches to and what callers still
reach through it.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from filelock import FileLock

# Re-exported: historical import location of these model helpers.
from nanoqc.model.model_egnn_pruning import (  # noqa: F401
    ModelLoadInfo,
    assert_checkpoint_graph_compatible,
)
from nanoqc.inference.paired_statistics import paired_effect as _paired_effect  # noqa: F401

# Names from the split-out modules that this module or its callers use.
from nanoqc.experiments.benchmark_common import (  # noqa: F401
    SHARED_HELPER_MODULES,
)
from nanoqc.experiments.hard_set_evaluation import (  # noqa: F401
    _build_parser,
    _run,
)
from nanoqc.experiments.research_ablation import (  # noqa: F401
    _ablation_classical_counts,
    queries_to_solution,
    load_transfer_parameters,
    _ablation_summarize,
    _ablation_run_case,
    _ablation_main,
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
