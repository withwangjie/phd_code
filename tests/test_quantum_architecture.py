"""Regression tests for the quantum-centered experiment architecture."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from nanoqc.quantum.instance import QuantumOptimizationInstance
from nanoqc.quantum.resource_estimation import estimate_qaoa_resources


REPO=Path(__file__).resolve().parents[1]


def _instance() -> QuantumOptimizationInstance:
    physical_self=np.asarray([1.0,2.0,3.0,4.0])
    physical_pair=np.zeros((4,4),dtype=float)
    physical_pair[0,2]=2.0
    return QuantumOptimizationInstance(
        Q=np.diag([1.0,2.0,3.0,4.0]),
        constant_offset=0.0,
        physical_self=physical_self,
        physical_pair=physical_pair,
        site_to_variables={0:(0,1),1:(2,3)},
        ising_h=np.zeros(4),
        ising_J=np.zeros((4,4)),
        ising_offset=0.0,
        metadata={"pdb_id":"test"},
    )


def test_quantum_instance_is_solver_facing_contract() -> None:
    instance=_instance()
    assert instance.num_qubits==4
    assert instance.register_sizes==(2,2)
    assert instance.feasible_configuration_count==4
    manifest=instance.manifest()
    assert manifest["schema"]=="quantum_optimization_instance_v1"
    assert manifest["encoding"]=="one_hot_rotamer_registers"
    assert manifest["full_qubo_includes_one_hot_penalties"] is True
    assert manifest["xy_cost_uses_penalty_free_feasible_subspace_objective"] is True


def test_penalty_free_physical_ising_matches_binary_energy() -> None:
    instance=_instance()
    h,j,offset=instance.physical_ising()
    for bits in (
        np.asarray([1,0,1,0],dtype=float),
        np.asarray([1,0,0,1],dtype=float),
        np.asarray([0,1,1,0],dtype=float),
        np.asarray([0,1,0,1],dtype=float),
    ):
        binary=float(instance.physical_self@bits+bits@instance.physical_pair@bits)
        spins=1.0-2.0*bits
        ising=float(offset+h@spins+spins@j@spins)
        assert np.isclose(binary,ising,atol=1e-12)


def test_qaoa_resource_estimate_matches_implemented_xy_ansatz() -> None:
    resources=estimate_qaoa_resources(
        _instance(),p=2,eval_shots=500,max_evals=90,output_shots=1000,restarts=4
    )
    assert resources.num_qubits==4
    assert resources.parameter_count==4
    assert resources.nonzero_pair_cost_terms==1
    assert resources.mixer_pairs_per_layer==2
    assert resources.zz_gates_total==2
    assert resources.xy_gates_total==4
    assert resources.two_qubit_gates_total==6
    assert resources.feasible_configuration_count==4
    assert resources.max_optimization_shots==45000
    assert resources.max_measurement_shots==46000


def test_benchmark_contract_separates_proposed_baselines_and_oracle() -> None:
    source=(REPO/"src/nanoqc/experiments/batch_benchmark_hard_set.py").read_text(encoding="utf-8")
    assert 'method_role="proposed_quantum_method"' in source
    assert 'method_role="classical_baseline"' in source
    assert '"schema":"quantum_classical_benchmark_v1"' in source
    assert '"role":"retrospective_ground_truth_only"' in source
    assert "quantum_instance={" in source
    assert "**quantum_instance.manifest()" in source
    assert '"Q":quantum_instance.Q.tolist()' in source
    assert "estimate_qaoa_resources(" in source


def test_qubo_result_exports_quantum_instance_contract() -> None:
    source=(REPO/"src/nanoqc/qubo/subgraph_to_qubo.py").read_text(encoding="utf-8")
    assert "def to_quantum_instance(self)" in source
    assert '"quantum_instance": quantum_path' in source


def test_quantum_protocol_is_single_source_of_truth() -> None:
    config=yaml.safe_load((REPO/"configs/full_experiment_config.yaml").read_text(encoding="utf-8"))
    qp=config["quantum_protocol"]
    assert qp["algorithm"]=="xy_qaoa"
    assert qp["mixer"]=="local_xy"
    assert qp["initial_state"]=="wstate"
    assert qp["primary"]["depth"]==2
    assert qp["primary"]["objective"]=="cvar"
    forbidden={"depths","max_evals","qaoa_objective","qaoa_restarts","cvar_alpha","eval_shots","parameter_scale"}
    assert not (forbidden & set(config["qc_benchmark"]))
    assert not ({"depths","max_evals","eval_shots","cvar_alpha","qaoa_objective","qaoa_restarts"} & set(config["qc_benchmark"]["sensitivity"]))
    assert not ({"outputs","max_evals","qaoa_depth","eval_shots","qaoa_restarts","qaoa_objective","cvar_alpha","parameter_scale"} & set(config["structure_experiment"]))
    assert not ({"primary_depth","primary_max_evals","primary_outputs","primary_objective","primary_restarts"} & set(config["statistics"]))


def test_scaling_reports_quantum_resource_axes_without_new_primary_tests() -> None:
    source=(REPO/"src/nanoqc/inference/analyze_quantum_scaling.py").read_text(encoding="utf-8")
    assert '"num_qubits","qaoa_two_qubit_gates","qaoa_xy_gates","qaoa_zz_gates"' in source
    assert "mean_qaoa_two_qubit_gates" in source
    assert 'primary_predictor="log10_configuration_count"' in source
    assert "descriptive_resource_axes" in source


def test_final_report_is_quantum_first() -> None:
    source=(REPO/"src/nanoqc/reporting/generate_final_research_report.py").read_text(encoding="utf-8")
    compile_block=source[source.index("def compile_report"):]
    order=[
        "section_quantum_problem_encoding(ctx)",
        "section_quantum_protocol(ctx)",
        "section_search_performance(ctx)",
        "section_structural_benefit(ctx)",
        "section_data_reliability(ctx)",
        "section_pruning_contribution(ctx)",
    ]
    positions=[compile_block.index(item) for item in order]
    assert positions==sorted(positions)


def test_formal_preflight_reuses_exact_scientific_resource_gate() -> None:
    source=(REPO/"scripts/formal_preflight.sh").read_text(encoding="utf-8")
    assert "_validate_scientific_config(config)" in source
    assert "orchestrator.stage_env_check()" in source
    assert "Formal environment/resource gate failed" in source


def test_quantum_instance_provenance_includes_rotamer_library_hash() -> None:
    source=(REPO/"src/nanoqc/qubo/subgraph_to_qubo.py").read_text(encoding="utf-8")
    assert "def _sha256_path(" in source
    assert '"rotamer_library_sha256": _sha256_path(self.rotamer_library_path)' in source
    assert '"rotamer_library_sha256": self.metadata.get("rotamer_library_sha256")' in source
