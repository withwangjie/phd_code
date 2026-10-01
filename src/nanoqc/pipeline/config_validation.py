"""Load and validate the frozen scientific configuration.

``load_config`` parses ``full_experiment_config.yaml`` and rejects any
protocol value outside its preregistered contract before a run starts.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover - reported, not silently swallowed
    raise SystemExit(
        "PyYAML is required (`pip install pyyaml`) to parse full_experiment_config.yaml. "
        f"Import failed: {exc}"
    )

from nanoqc.pipeline.orchestrator_common import (
    MAX_QAOA_DEPTH,
    quantum_benchmark_ablation,
    quantum_development_sensitivity,
    quantum_primary,
    quantum_protocol,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _validate_scientific_config(config: Dict[str, Any]) -> None:
    """Fail before any stage when scientific protocol settings are inconsistent."""

    qf = config.get("queue_freeze", {}) or {}
    graph = qf.get("graph_build", {}) or {}
    # A20: the formal EGNN trains in FP32. FP16 autocast overflowed on
    # coordinate and squared-distance terms (non-finite loss on T4), so the key
    # must be present and exactly false; a missing key is not accepted because
    # the stage would otherwise fall back to a default.
    egnn = config.get("egnn_train", {}) or {}
    if "amp" not in egnn or egnn["amp"] is not False:
        raise ValueError(
            "egnn_train.amp must be explicitly false: formal EGNN training is FP32 "
            f"(PROTOCOL_AMENDMENTS.md A20); got {egnn.get('amp', '<missing>')!r}")
    qc = config.get("qc_benchmark", {}) or {}
    structure = config.get("structure_experiment", {}) or {}
    qproto = quantum_protocol(config)
    qprimary = quantum_primary(config)
    qablation = quantum_benchmark_ablation(config)
    qsensitivity = quantum_development_sensitivity(config)
    validation = qf.get("validation_queue", {}) or {}
    clustering = qf.get("independence_clustering", {}) or {}
    if str(qproto.get("algorithm","")) != "xy_qaoa":
        raise ValueError("quantum_protocol.algorithm must be xy_qaoa")
    if str(qproto.get("encoding","")) != "one_hot_rotamer_registers":
        raise ValueError("quantum_protocol.encoding must be one_hot_rotamer_registers")
    if str(qproto.get("mixer","")) != "local_xy":
        raise ValueError("quantum_protocol.mixer must be local_xy")
    if str(qproto.get("initial_state","")) != "wstate":
        raise ValueError(
            "quantum_protocol.initial_state must be wstate (product of local one-hot W states)"
        )
    if str(qproto.get("simulation_scope","")) != "exact_feasible_subspace_classical_simulation":
        raise ValueError(
            "quantum_protocol.simulation_scope must explicitly identify exact feasible-subspace classical simulation"
        )

    legacy_qc_keys={
        "depths","max_evals","qaoa_objective","qaoa_restarts",
        "cvar_alpha","eval_shots","parameter_scale",
    }
    stale_qc=sorted(k for k in legacy_qc_keys if k in qc)
    if stale_qc:
        raise ValueError(
            f"QAOA settings must live only under quantum_protocol; remove qc_benchmark keys {stale_qc}"
        )
    legacy_structure_keys={
        "outputs","max_evals","qaoa_depth","eval_shots","qaoa_restarts",
        "qaoa_objective","cvar_alpha","parameter_scale",
    }
    stale_structure=sorted(k for k in legacy_structure_keys if k in structure)
    if stale_structure:
        raise ValueError(
            f"QAOA settings must live only under quantum_protocol; remove structure_experiment keys {stale_structure}"
        )
    stats_probe=config.get("statistics",{}) or {}
    legacy_stats_keys={
        "primary_depth","primary_max_evals","primary_outputs",
        "primary_objective","primary_restarts",
    }
    stale_stats=sorted(k for k in legacy_stats_keys if k in stats_probe)
    if stale_stats:
        raise ValueError(
            f"QAOA primary settings must live only under quantum_protocol.primary; remove statistics keys {stale_stats}"
        )

    depth=int(qprimary.get("depth",0) or 0)
    if not 1<=depth<=MAX_QAOA_DEPTH:
        raise ValueError(f"quantum_protocol.primary.depth must be in 1..{MAX_QAOA_DEPTH}")
    for key in ("max_evals","restarts","eval_shots","output_shots"):
        value=qprimary.get(key)
        if isinstance(value,bool) or value is None or int(value)!=value or int(value)<=0:
            raise ValueError(f"quantum_protocol.primary.{key} must be a positive integer")
    if str(qprimary.get("objective","")) not in ("mean","cvar"):
        raise ValueError("quantum_protocol.primary.objective must be mean or cvar")
    alpha=float(qprimary.get("cvar_alpha",0.0))
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("quantum_protocol.primary.cvar_alpha must lie in (0,1]")
    if str(qprimary.get("parameter_scale","")) not in ("max_coefficient","feasible_iqr"):
        raise ValueError(
            "quantum_protocol.primary.parameter_scale must be max_coefficient or feasible_iqr"
        )

    objectives=[str(v) for v in qablation.get("objectives",[])]
    restarts=[int(v) for v in qablation.get("restarts",[])]
    if not objectives or any(v not in ("mean","cvar") for v in objectives) or len(objectives)!=len(set(objectives)):
        raise ValueError("quantum_protocol.benchmark_ablation.objectives must be unique mean/cvar values")
    if not restarts or any(v<=0 for v in restarts) or len(restarts)!=len(set(restarts)):
        raise ValueError("quantum_protocol.benchmark_ablation.restarts must be unique positive integers")
    if str(qprimary["objective"]) not in objectives or int(qprimary["restarts"]) not in restarts:
        raise ValueError("quantum_protocol primary objective/restarts must be present in benchmark_ablation")

    qc_sensitivity=qc.get("sensitivity",{}) or {}
    stale_nested=sorted(
        k for k in ("depths","max_evals","eval_shots","cvar_alpha","qaoa_objective","qaoa_restarts")
        if k in qc_sensitivity
    )
    if stale_nested:
        raise ValueError(
            f"Quantum sensitivity axes must live only under quantum_protocol.development_sensitivity; "
            f"remove qc_benchmark.sensitivity keys {stale_nested}"
        )
    if structure.get("robust_qaoa",True) is not True:
        raise ValueError("Formal structure_experiment.robust_qaoa must remain true for the frozen quantum protocol")

    sensitivity_depths=[int(v) for v in qsensitivity.get("depths",[])]
    if not sensitivity_depths or any(not 1<=v<=MAX_QAOA_DEPTH for v in sensitivity_depths):
        raise ValueError(f"quantum_protocol.development_sensitivity.depths must use supported p in 1..{MAX_QAOA_DEPTH}")
    for key in ("max_evals","eval_shots"):
        values=[int(v) for v in qsensitivity.get(key,[])]
        if not values or any(v<=0 for v in values) or len(values)!=len(set(values)):
            raise ValueError(f"quantum_protocol.development_sensitivity.{key} must be unique positive integers")
    sensitivity_alpha=[float(v) for v in qsensitivity.get("cvar_alpha",[])]
    if (not sensitivity_alpha or any((not math.isfinite(v)) or not 0.0<v<=1.0 for v in sensitivity_alpha)
            or len(sensitivity_alpha)!=len(set(sensitivity_alpha))):
        raise ValueError("quantum_protocol.development_sensitivity.cvar_alpha must be unique values in (0,1]")
    if int(qprimary["depth"]) not in sensitivity_depths:
        raise ValueError("quantum primary depth must be included in development_sensitivity.depths")
    if int(qprimary["max_evals"]) not in [int(v) for v in qsensitivity.get("max_evals",[])]:
        raise ValueError("quantum primary max_evals must be included in development_sensitivity.max_evals")
    if int(qprimary["eval_shots"]) not in [int(v) for v in qsensitivity.get("eval_shots",[])]:
        raise ValueError("quantum primary eval_shots must be included in development_sensitivity.eval_shots")
    if float(qprimary["cvar_alpha"]) not in sensitivity_alpha:
        raise ValueError("quantum primary cvar_alpha must be included in development_sensitivity.cvar_alpha")

    exploration=config.get("quantum_exploration",{}) or {}
    if exploration:
        exploration_depths=[int(v) for v in exploration.get("depths",[])]
        if (not exploration_depths or len(set(exploration_depths))!=len(exploration_depths)
                or any(not 1<=v<=MAX_QAOA_DEPTH for v in exploration_depths)):
            raise ValueError(f"quantum_exploration.depths must be unique integers in 1..{MAX_QAOA_DEPTH}")
        per_parameter=int(exploration.get("evals_per_parameter",0) or 0)
        restarts_primary=int(qprimary.get("restarts",4))
        if per_parameter<=0 or any(per_parameter*2*d < 1+restarts_primary*(2*d+2) for d in exploration_depths):
            raise ValueError("quantum_exploration.evals_per_parameter is too small for the primary restarts")
        if int(exploration.get("repeats",0) or 0)<=0:
            raise ValueError("quantum_exploration.repeats must be positive")
        transfer=exploration.get("transfer",{}) or {}
        if int(transfer.get("train_max_targets",0) or 0)<int(transfer.get("min_instances",1) or 1):
            raise ValueError("quantum_exploration.transfer.train_max_targets must be >= min_instances")

    if clustering.get("required", False) and not clustering.get("cluster_map"):
        raise ValueError("queue_freeze.independence_clustering.cluster_map is required")

    homology = qf.get("homology_isolation", {}) or {}
    required_homology = {
        "vhh_full_chain_identity": float(homology.get("vhh_full_chain_identity", 0.80)),
        "cdr_h3_identity": float(homology.get("cdr_h3_identity", 0.50)),
        "antigen_identity": float(homology.get("antigen_identity", 0.30)),
        "antigen_min_length_coverage": float(homology.get("antigen_min_length_coverage", 0.70)),
    }
    if any(not 0.0 < value <= 1.0 for value in required_homology.values()):
        raise ValueError(f"homology_isolation values must lie in (0,1]: {required_homology}")

    positive_graph = {
        "interface_label_cutoff_angstrom": float(graph.get("interface_label_cutoff_angstrom", 4.5)),
        "intra_chain_ca_cutoff_angstrom": float(graph.get("intra_chain_ca_cutoff_angstrom", 8.0)),
    }
    if any((not math.isfinite(v)) or v <= 0 for v in positive_graph.values()):
        raise ValueError(f"Graph distance parameters must be positive finite: {positive_graph}")
    if int(graph.get("cross_partner_knn_k", 3)) < 1:
        raise ValueError("graph_build.cross_partner_knn_k must be >=1")
    if int(graph.get("min_interface_residues", 15)) < 1:
        raise ValueError("graph_build.min_interface_residues must be >=1")
    primary_interface=float(graph.get("interface_label_cutoff_angstrom",4.5))
    sensitivity_interfaces=[float(v) for v in graph.get(
        "interface_sensitivity_cutoffs_angstrom",[3.5,5.0])]
    if not math.isclose(primary_interface,4.5,rel_tol=0.0,abs_tol=1e-12):
        raise ValueError("Formal primary interface_label_cutoff_angstrom must remain 4.5 A")
    if sensitivity_interfaces != [3.5,5.0]:
        raise ValueError("Formal interface sensitivity cutoffs must remain [3.5, 5.0] A")
    audit_interface=float((config.get("data_audit",{}) or {}).get(
        "interface_contact_cutoff_angstrom",4.5))
    minimum_heavy_distance=float((config.get("data_audit",{}) or {}).get(
        "min_interresidue_heavy_distance_angstrom",1.0))
    if not math.isfinite(minimum_heavy_distance) or not 0<minimum_heavy_distance<=1.0:
        raise ValueError("data_audit.min_interresidue_heavy_distance_angstrom must be in (0,1.0]")
    if not math.isclose(audit_interface,primary_interface,rel_tol=0.0,abs_tol=1e-12):
        raise ValueError("data_audit and graph_build primary interface cutoffs must match")
    if int(structure.get("loop_relax_iterations",0) or 0) != 0:
        raise ValueError(
            "Formal structure_experiment is strict fixed-backbone: loop_relax_iterations must be 0")

    if not 0.0 <= float(qc.get("antigen_guidance_weight", 0.25)) <= 1.0:
        raise ValueError("qc_benchmark.antigen_guidance_weight must be in [0,1]")
    if not 0.0 <= float(structure.get("antigen_guidance_weight", 0.25)) <= 1.0:
        raise ValueError("structure_experiment.antigen_guidance_weight must be in [0,1]")

    shared_pairs = (
        ("antigen_guidance_weight", 0.25),
        ("antigen_proximity_scale_angstrom", 6.0),
        ("contact_ca_cutoff_angstrom", 8.0),
    )
    qc_rot = qc.get("rotamer_model", {}) or {}
    st_rot = structure.get("rotamer_model", {}) or {}
    rotamer_keys = ("mode", "library_path", "version_contains", "probability_floor", "sigma_offsets")
    for key in rotamer_keys:
        if qc_rot.get(key) != st_rot.get(key):
            raise ValueError(
                f"Rotamer protocol mismatch for {key}: "
                f"qc_benchmark={qc_rot.get(key)!r}, structure_experiment={st_rot.get(key)!r}"
            )
    if qc_rot.get("mode", "dunbrack2010") in ("dunbrack2010", "pyrosetta_dun10"):
        floor = float(qc_rot.get("probability_floor", 1e-4))
        offsets = qc_rot.get("sigma_offsets", [-1.0, 0.0, 1.0])
        if not 0.0 < floor < 1.0 or not offsets:
            raise ValueError("Invalid Dunbrack probability floor or sigma offsets")
        if any(not math.isfinite(float(v)) for v in offsets):
            raise ValueError("Dunbrack sigma offsets must be finite")

    if clustering.get("required", False):
        pair_tsv=clustering.get("pair_tsv")
        if not clustering.get("cluster_map"):
            raise ValueError("independence_clustering.cluster_map is required")
        if clustering.get("build_per_run",False) and not pair_tsv:
            raise ValueError("run-local Foldseek pairs require independence_clustering.pair_tsv")
        if clustering.get("build_per_run",False) and str(clustering.get("score_semantics","")) != "mintmscore":
            raise ValueError("run-local Foldseek pairs require score_semantics=mintmscore")
        if pair_tsv is not None:
            min_score=float(clustering.get("min_score",0.50))
            if not math.isfinite(min_score):
                raise ValueError("independence_clustering.min_score must be finite")
            if str(clustering.get("score_semantics","")).lower() not in ("qtmscore","ttmscore","mintmscore"):
                raise ValueError("independence_clustering.score_semantics must be qtmscore, ttmscore or mintmscore")
            for key in ("query_column","target_column","score_column"):
                if int(clustering.get(key,0)) < 0:
                    raise ValueError(f"independence_clustering.{key} must be nonnegative")

    solvent_model=str(structure.get("solvent_model","vacuum")).lower()
    if solvent_model not in ("vacuum","gbn2"):
        raise ValueError("structure_experiment.solvent_model must be vacuum or gbn2")
    sensitivity=[str(v).lower() for v in structure.get("solvent_sensitivity",[])]
    if any(v not in ("vacuum","gbn2") for v in sensitivity):
        raise ValueError("structure_experiment.solvent_sensitivity supports only vacuum/gbn2")
    if len(sensitivity)!=len(set(sensitivity)):
        raise ValueError("structure_experiment.solvent_sensitivity must not contain duplicates")
    if str(structure.get("perturbation_mode","multi_chi")) not in ("multi_chi","chi1"):
        raise ValueError("structure_experiment.perturbation_mode must be multi_chi or chi1")
    attempts = structure.get("perturbation_max_attempts", 32)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("structure_experiment.perturbation_max_attempts must be a positive integer")
    seeds=[int(v) for v in structure.get("seeds",[42,43,44,45,46])]
    if len(seeds)<3 or len(seeds)!=len(set(seeds)) or min(seeds)<0:
        raise ValueError("structure_experiment.seeds must contain >=3 unique nonnegative values")
    if int(qc.get("repeats",10)) < 3:
        raise ValueError("qc_benchmark.repeats must be >=3")
    max_failure_fraction=float(qc.get("max_failure_fraction",0.0))
    if not math.isfinite(max_failure_fraction) or not 0.0 <= max_failure_fraction < 1.0:
        raise ValueError("qc_benchmark.max_failure_fraction must be in [0,1)")

    stats=config.get("statistics",{}) or {}
    if stats.get("cluster_map") not in (None, ""):
        raise ValueError(
            "statistics.cluster_map is obsolete: formal statistics must reuse the "
            "run-local frozen cluster map produced by queue_freeze; remove the key"
        )
    if stats.get("primary_structural_endpoint","final_rmsd") not in (
        "final_rmsd","improvement_vs_input","improvement_vs_relax_only"
    ):
        raise ValueError("Invalid statistics.primary_structural_endpoint")
    if stats.get("primary_structural_contrast","qaoa_vs_sa") not in (
        "qaoa_vs_sa","qaoa_vs_greedy","qaoa_vs_uniform"
    ):
        raise ValueError("Invalid statistics.primary_structural_contrast")
    if int(stats.get("resamples",10000)) < 1000:
        raise ValueError("statistics.resamples must be >=1000")
    if int(stats.get("min_qc_clusters",10)) < 2:
        raise ValueError("statistics.min_qc_clusters must be >=2")
    if int(stats.get("min_scaling_clusters",10)) < 2:
        raise ValueError("statistics.min_scaling_clusters must be >=2")
    if str(stats.get("primary_qc_baseline","sa")) not in ("sa","uniform","greedy"):
        raise ValueError("statistics.primary_qc_baseline must be sa, uniform, or greedy")
    if str(stats.get("primary_qc_metric","log10_qts99")) not in (
        "gap","hit","ground_probability","low_energy_mass","low_energy_coverage","entropy","log10_qts99"
    ):
        raise ValueError("Invalid statistics.primary_qc_metric")
    if int(stats.get("min_primary_clusters",10)) < 2:
        raise ValueError("statistics.min_primary_clusters must be >=2")
    if int(stats.get("min_rq5_clusters",10)) < 2:
        raise ValueError("statistics.min_rq5_clusters must be >=2")

    external=config.get("external_validation",{}) or {}
    if external.get("required",False):
        ext=external.get("external_vhh",{}) or {}
        structural=external.get("structural_baselines",{}) or {}
        # Empty directories mean this run's own antigen-fold holdout, resolved
        # against the run directory once the dataset exists (A10). Configuring
        # one without the other is always a mistake.
        if ext.get("required",False) and bool(ext.get("graph_dir")) != bool(ext.get("source_structure_dir")):
            raise ValueError("external_validation.external_vhh.graph_dir and source_structure_dir must be "
                             "set together, or both left empty to score this run's antigen-fold holdout")
        if ext.get("independence_manifest"):
            # The independence audit is bound to this run's frozen dataset and
            # cluster map, so it is always regenerated inside the run directory.
            raise ValueError(
                "external_validation.external_vhh.independence_manifest is obsolete: the audit "
                "is regenerated as <run_dir>/external_validation/external_vhh_independence_manifest.json; "
                "remove the key")
        if structural.get("required",False):
            if not structural.get("faspr_executable") or not structural.get("phenix_clashscore_executable"):
                raise ValueError("Required structural baseline executables must be configured")

    calibration_cfg = qc.get("energy_calibration", {}) or {}
    mode = str(calibration_cfg.get("mode", "frozen"))
    if mode not in ("frozen", "diagnostic", "off"):
        raise ValueError("energy_calibration.mode must be frozen, diagnostic, or off")
    if mode == "diagnostic" and calibration_cfg.get("require_calibrated", False):
        raise ValueError("Diagnostic calibration must not be applied to solver benchmarks")
    ridge_alpha = float(calibration_cfg.get("ridge_alpha", 1.0))
    if not math.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("energy_calibration.ridge_alpha must be finite and nonnegative")
    quality_limit=(calibration_cfg.get("acceptance",{}) or {}).get(
        "max_input_quality_exclusion_fraction")
    if quality_limit is not None and (
            not math.isfinite(float(quality_limit)) or not 0.0 <= float(quality_limit) < 1.0):
        raise ValueError("energy_calibration.acceptance.max_input_quality_exclusion_fraction must be null or in [0,1)")

    for key, default in shared_pairs:
        left = float(qc.get(key, default))
        right = float(structure.get(key, default))
        if key != "antigen_guidance_weight" and (
            not math.isfinite(left) or left <= 0 or not math.isfinite(right) or right <= 0
        ):
            raise ValueError(f"{key} must be positive finite in coarse and structure protocols")
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Shared site-selection parameter mismatch for {key}: "
                f"qc_benchmark={left}, structure_experiment={right}"
            )

    qc_sites=[int(v) for v in qc.get("active_sites",[6])]
    if not qc_sites or len(qc_sites)!=len(set(qc_sites)) or any(v<4 or v>10 for v in qc_sites):
        raise ValueError("qc_benchmark.active_sites must contain unique integers in 4..10")
    resolution=(qc.get("sensitivity",{}) or {}).get("rotamer_resolution",{}) or {}
    if resolution:
        site_levels=[int(v) for v in resolution.get("active_sites",[])]
        state_levels=[int(v) for v in resolution.get("states_per_site",[])]
        if (not site_levels or not state_levels or len(site_levels)!=len(set(site_levels))
                or len(state_levels)!=len(set(state_levels))
                or any(site not in (4,5) for site in site_levels)
                or any(state not in (3,4,5,6) for state in state_levels)
                or any(site*state>30 for site in site_levels for state in state_levels)):
            raise ValueError("rotamer_resolution requires unique 4-5 sites and 3-6 states/site within 30 bits")
    primary_pruning=str(stats.get("primary_pruning","egnn"))
    if primary_pruning not in [str(v) for v in qc.get("pruning",["egnn"])]:
        raise ValueError(
            f"statistics.primary_pruning={primary_pruning!r} must be present in "
            f"qc_benchmark.pruning={qc.get('pruning')}")
    primary_radius=float(stats.get("primary_radius",6.0))
    if primary_radius not in [float(v) for v in qc.get("radii",[6.0])]:
        raise ValueError("statistics.primary_radius must be present in qc_benchmark.radii")
    if int(qprimary["output_shots"]) not in [int(v) for v in qc.get("outputs",[1000])]:
        raise ValueError(
            "quantum_protocol.primary.output_shots must be present in qc_benchmark.outputs"
        )
    stats_primary_sites=int(stats.get("primary_active_sites",6))
    if stats_primary_sites not in qc_sites:
        raise ValueError(
            f"statistics.primary_active_sites={stats_primary_sites} must be present in "
            f"qc_benchmark.active_sites={qc_sites}")
    validation_sites = int(validation.get("sites", 6))
    if validation_sites != stats_primary_sites:
        raise ValueError(
            f"Formal structural active-site count must match the primary confirmatory size: "
            f"statistics.primary_active_sites={stats_primary_sites}, "
            f"validation_queue={validation_sites}"
        )

    ff = qc.get("coarse_force_field", {}) or {}
    positive_ff = {
        "cutoff_angstrom": float(ff.get("cutoff_angstrom", 8.0)),
        "softcore_delta_angstrom": float(ff.get("softcore_delta_angstrom", 0.5)),
        "hard_core_fraction": float(ff.get("hard_core_fraction", 0.72)),
        "hard_sphere_penalty": float(ff.get("hard_sphere_penalty", 25.0)),
        "lj_repulsion_cap": float(ff.get("lj_repulsion_cap", 50.0)),
        "lj_attraction_cap": float(ff.get("lj_attraction_cap", 5.0)),
        "coulomb_cap": float(ff.get("coulomb_cap", 20.0)),
        "dielectric_base": float(ff.get("dielectric_base", 4.0)),
        "thermal_energy_kcal": float(ff.get("thermal_energy_kcal", 0.593)),
    }
    if any((not math.isfinite(v)) or v <= 0 for v in positive_ff.values()):
        raise ValueError(f"coarse_force_field positive parameters invalid: {positive_ff}")
    dielectric_slope = float(ff.get("dielectric_slope", 2.0))
    if not math.isfinite(dielectric_slope) or dielectric_slope < 0:
        raise ValueError("coarse_force_field.dielectric_slope must be finite and nonnegative")


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a top-level YAML mapping")
    _validate_scientific_config(config)
    return config
