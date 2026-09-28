"""Code-fingerprint list shared by every benchmark mode's provenance record.
"""
from __future__ import annotations


# Shared helper modules (factored-out helpers plus the formal quantum instance/resource modules) imported by the modules
# fingerprinted below; they are hashed alongside them in every code_sha256.
SHARED_HELPER_MODULES = ("repo_io.py", "paired_statistics.py", "residue_tables.py", "safe_graph_load.py",
                         "instance.py", "resource_estimation.py",
                         # subgraph_to_qubo.py re-exports these; the QUBO code itself lives here.
                         "atomistic_structure.py",
                         "qubo_types.py",
                         "rotamer_library.py",
                         "coarse_qubo.py",
                         "ising.py",
                         "allatom_qubo.py",
                         # batch_benchmark_hard_set.py dispatches; the modes themselves live here.
                         "benchmark_common.py",
                         "hard_set_evaluation.py",
                         "research_ablation.py",
                         "calibration_fit.py",
                         "benchmark_statistics.py",
                         "structure_benchmarks.py",
                         )
