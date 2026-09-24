"""Regression tests for the high-severity audit fixes (resume, homology roles, GBN2)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

import nanoqc.data.audit_external_vhh_independence as external_audit
import nanoqc.data.build_final_pyg_dataset as dataset_builder
import nanoqc.experiments.batch_benchmark_hard_set as bbh
import nanoqc.experiments.generate_energy_calibration_dataset as calibration
import nanoqc.experiments.run_real_complex_pilot as pilot
import nanoqc.model.train_egnn_pruning as train
import nanoqc.pipeline.run_full_experiment as full
import nanoqc.qubo.subgraph_to_qubo as qubo
from nanoqc.data.sequence_identity import partner_orientations, partner_roles_anchored
from nanoqc.pipeline.run_full_experiment import Orchestrator, StageResult


# ---------------------------------------------------------------- resume logic
def _orchestrator(tmp_path: Path) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.run_dir = tmp_path
    orchestrator.only = None
    orchestrator.force_restage = set()
    orchestrator.config = {"stages": {}}
    orchestrator.status_dir = tmp_path / "stage_status"
    orchestrator.results_manifest_dir = tmp_path / "results_manifests"
    orchestrator.log_dir = tmp_path / "logs"
    orchestrator.progress_path = tmp_path / "progress.json"
    for folder in (orchestrator.status_dir, orchestrator.results_manifest_dir, orchestrator.log_dir):
        folder.mkdir(parents=True, exist_ok=True)
    orchestrator._stage_result_roots = lambda stage: []
    orchestrator._validate_completed_stage_artifacts = (
        lambda stage, require_results_manifest=True: (True, "ok"))
    return orchestrator


def _completed(stage: str) -> StageResult:
    return StageResult(stage, "completed", "a", "b", 0, "ok")


def test_prerequisite_blocked_stage_is_retried_once_prerequisites_complete(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    calls = []

    def downstream():
        calls.append("ran")
        return _completed("downstream")

    blocked = orchestrator.run_stage("downstream", ["upstream"], downstream)
    assert blocked.status == "failed" and blocked.detail.startswith(full.PREREQUISITE_BLOCK_PREFIX)
    assert calls == []
    orchestrator.run_stage("upstream", [], lambda: _completed("upstream"))
    result = orchestrator.run_stage("downstream", ["upstream"], downstream)
    assert result.status == "completed" and calls == ["ran"]


def test_genuine_failure_is_still_not_auto_retried(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.run_stage("stage", [], lambda: StageResult("stage", "failed", "a", "b", 1, "boom"))
    calls = []
    result = orchestrator.run_stage("stage", [], lambda: calls.append(1) or _completed("stage"))
    assert result.status == "failed" and result.detail.startswith("Not retried") and calls == []


def test_downstream_results_are_rejected_after_upstream_rerun(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    artifact = tmp_path / "upstream" / "result.txt"
    orchestrator._stage_result_roots = lambda stage: [tmp_path / "upstream"] if stage == "upstream" else []

    def upstream(content):
        def run():
            artifact.parent.mkdir(exist_ok=True)
            artifact.write_text(content, encoding="utf-8")
            return _completed("upstream")
        return run

    orchestrator.run_stage("upstream", [], upstream("v1"))
    orchestrator.run_stage("downstream", ["upstream"], lambda: _completed("downstream"))
    calls = []
    resumed = orchestrator.run_stage("downstream", ["upstream"],
                                     lambda: calls.append(1) or _completed("downstream"))
    assert resumed.status == "completed" and resumed.detail.startswith("Resumed") and calls == []

    # A bit-identical upstream re-run keeps downstream results valid ...
    orchestrator.force_restage = {"upstream"}
    orchestrator.run_stage("upstream", [], upstream("v1"))
    orchestrator.force_restage = set()
    still_valid = orchestrator.run_stage("downstream", ["upstream"],
                                         lambda: calls.append(1) or _completed("downstream"))
    assert still_valid.status == "completed" and calls == []
    # ... a changed upstream artifact does not.
    orchestrator.force_restage = {"upstream"}
    orchestrator.run_stage("upstream", [], upstream("v2"))
    orchestrator.force_restage = set()
    stale = orchestrator.run_stage("downstream", ["upstream"],
                                   lambda: calls.append(1) or _completed("downstream"))
    assert stale.status == "failed" and "upstream stage results changed" in stale.detail
    assert calls == []


def test_graph_build_records_resume_identity_before_work_starts():
    source = inspect.getsource(dataset_builder.main)
    assert source.index("summary['resume_identity']=resume_identity") < source.index("rows,pairs,input_hashes=load_inputs")
    assert "prior.get('resume_identity')" in source


# ------------------------------------------------------ partner-role homology
def test_partner_orientations_cover_swapped_unanchored_roles():
    nanobody, lysozyme = ("QVQLVESGG",), ("KVFGRCELAA",)
    target = (nanobody, lysozyme)          # anchored: VHH, antigen
    swapped_training = (lysozyme, nanobody)  # train_rcsb with roles reversed
    orientations = list(partner_orientations(*target, True, *swapped_training, False))
    assert (nanobody, nanobody, lysozyme, lysozyme) in orientations
    anchored_only = list(partner_orientations(*target, True, *target, True))
    assert anchored_only == [(nanobody, nanobody, lysozyme, lysozyme)]
    assert partner_roles_anchored("snac_db") and partner_roles_anchored("sabdab_vhh")
    assert not partner_roles_anchored("train_rcsb") and not partner_roles_anchored("")


def test_every_homology_consumer_uses_partner_orientations():
    assert "partner_orientations" in inspect.getsource(dataset_builder.layered_graph_homology)
    assert "partner_orientations" in inspect.getsource(train.split_paths)
    assert "partner_orientations" in inspect.getsource(external_audit.max_training_identities)
    assert "max_training_identities" in inspect.getsource(external_audit.main)
    assert "partner_roles_anchored" in inspect.getsource(pilot.main)



def _write_split_graph(path: Path, vhh: str, antigen: str, cdr3: str, family: str, source: str) -> Path:
    import torch
    from torch_geometric.data import Data
    graph = Data(x=torch.zeros(1, 3))
    graph.vhh_sequences = [vhh]
    graph.antigen_sequences = [antigen]
    graph.cdr3_seq = cdr3
    graph.family_structure_cluster = family
    graph.subset_source = source
    torch.save(graph, path)
    return path


def test_split_paths_runs_end_to_end_and_keeps_swapped_roles_together(tmp_path):
    import random
    rng = random.Random(7)
    residues = "ACDEFGHIKLMNPQRSTVWY"

    def sequence(length: int) -> str:
        return "".join(rng.choice(residues) for _ in range(length))

    paths = []
    for index in range(12):
        paths.append(_write_split_graph(
            tmp_path / f"g{index:02d}.pt", sequence(120), sequence(200), sequence(12),
            f"fam{index}", "sabdab_vhh"))
    # An unanchored complex whose partners are stored swapped relative to g00.
    import torch
    first = torch.load(paths[0], weights_only=False)
    swapped = _write_split_graph(
        tmp_path / "swapped.pt", first.antigen_sequences[0], first.vhh_sequences[0],
        "", "fam_swapped", "train_rcsb")
    paths.append(swapped)

    train_paths, validation_paths = train.split_paths(paths, 0)
    assert train_paths and validation_paths
    assert set(train_paths) | set(validation_paths) == set(paths)
    assert not set(train_paths) & set(validation_paths)
    assert (paths[0] in train_paths) == (swapped in train_paths)


def test_dataset_isolation_checks_survive_python_optimize_mode():
    import ast
    import pytest
    tree = ast.parse(inspect.getsource(dataset_builder))
    assert not [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    with pytest.raises(AssertionError, match="overlap"):
        dataset_builder._require(False, "train/test overlap")
    dataset_builder._require(True, "unused")


def test_measured_identities_name_full_chain_versus_cdr_h3_loop():
    from torch_geometric.data import Data
    left, right = Data(), Data()
    for graph, vhh, antigen, cdr in ((left, "QVQLVESGGGLVQ", "KVFGRCELAAAMK", "CARDRST"),
                                     (right, "EVQLLESGGGLVQ", "KVFGRCELAAAMR", "CAKDRSY")):
        graph.vhh_sequences, graph.antigen_sequences, graph.cdr3_seq = [vhh], [antigen], cdr
        graph.subset_source = "sabdab_vhh"
    detail = dataset_builder.layered_graph_homology(left, right)
    assert {"vhh_full_chain_identity", "cdr_h3_loop_identity", "antigen_full_chain_identity"} <= set(detail)
    assert not {"vhh_identity", "cdr_h3_identity", "antigen_identity"} & set(detail)
    assert detail["cdr_h3_loop_identity"] == dataset_builder.cdr_h3_loop_seqsim("CARDRST", "CAKDRSY")
    for module in (pilot, external_audit):
        source = inspect.getsource(module)
        assert "max_vhh_full_chain_identity" in source
        assert "max_cdr_h3_loop_identity" in source
        assert "max_vhh_identity" not in source and "cdr3_identity=" not in source

# ------------------------------------------------ GBN2 / calibration / external
def test_gbn2_is_a_recorded_pairwise_approximation_and_vacuum_stays_exact():
    build = inspect.getsource(qubo.AllAtomInterfaceQUBOBuilder.build)
    assert "if exact and not np.isclose(actual,predicted,atol=1e-4,rtol=1e-9)" in build
    assert '"exact" if exact else "pairwise_approximation"' in build
    init = inspect.getsource(qubo.AllAtomInterfaceQUBOBuilder.__init__)
    assert 'self.pair_decomposition_exact=self.solvent_model=="vacuum"' in init
    experiment = inspect.getsource(bbh._allatom_experiment_main)
    assert "qubo_energy_discrepancy_kcal" in experiment


def test_calibration_uses_the_target_force_field_preparation():
    source = inspect.getsource(calibration.main)
    assert "strip_to_protein_conformer(structure,residues)" in source
    assert "strip_to_protein_conformer(st, residues)" in inspect.getsource(pilot.prepare)


def test_external_independence_manifest_is_run_local_and_regenerated():
    source = inspect.getsource(Orchestrator.stage_external_validation)
    assert 'independence=run_external_root/"external_vhh_independence_manifest.json"' in source
    assert "independence.unlink()" in source
    import copy
    import yaml
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" /
                             "full_experiment_config.yaml").read_text(encoding="utf-8"))
    full._validate_scientific_config(copy.deepcopy(config))
    config["external_validation"]["external_vhh"]["independence_manifest"] = "x.json"
    try:
        full._validate_scientific_config(config)
    except ValueError as exc:
        assert "independence_manifest is obsolete" in str(exc)
    else:
        raise AssertionError("a repo-level independence manifest must be rejected")


# ------------------------------------------- full-complex antigen environment
def _graph_with_antigen_outside_radius():
    import torch
    data = qubo._virtual_pruned_graph(5)
    extra_x = torch.zeros((1, 21))
    extra_x[0, qubo.AA_INDEX["K"]] = 1.0
    extra_x[0, -1] = 1.0
    # Charged antigen residue whose CA is > 6 A from every Active CA (so the
    # 6 A environment radius drops it) but well inside cutoff + side-chain reach.
    # A charged VHH residue in the same situation for the fixed-VHH term.
    vhh_x = torch.zeros((1, 21))
    vhh_x[0, qubo.AA_INDEX["E"]] = 1.0
    data.x = torch.cat([data.x, extra_x, vhh_x])
    data.pos = torch.cat([data.pos, torch.tensor([[-4.3, -4.6, 0.0], [-3.0, -5.6, 0.0]])])
    distances = torch.cdist(data.pos, data.pos)
    data.edge_index = ((distances < 8.0) & (distances > 0)).nonzero().t().long()
    data.node_chain_id = torch.cat([data.node_chain_id, torch.tensor([1, 0])])
    data.interface_score = torch.cat([data.interface_score, torch.tensor([0.05, 0.04])])
    data.residue_ids = [*data.residue_ids, "A:99", "H:99"]
    for name in ("is_active", "is_frozen_environment", "selected_vhh_mask", "original_node_index"):
        delattr(data, name)
    return data, data.num_nodes - 2


def _coarse_builder():
    return qubo.InterfaceQUBOBuilder(min_variables=15, max_variables=30, max_sites=5)


def test_antigen_environment_uses_full_complex_not_the_environment_radius():
    import copy
    import torch
    from nanoqc.model.model_egnn_pruning import build_ablation_subgraph
    data, far = _graph_with_antigen_outside_radius()
    active = torch.arange(5)
    assert torch.cdist(data.pos[far:far + 1], data.pos[active]).min() > 6.0
    sub = build_ablation_subgraph(data, active, 6.0)
    assert far not in sub.original_node_index.tolist()
    assert len(sub.antigen_context_pos) == int((data.x[:, -1] == 1).sum())

    full = _coarse_builder().build(sub)
    truncated_sub = copy.copy(sub)
    for name in ("antigen_context_pos", "antigen_context_x", "vhh_context_pos",
                 "vhh_context_x", "vhh_context_index"):
        delattr(truncated_sub, name)
    truncated = _coarse_builder().build(truncated_sub)
    assert full.metadata["antigen_environment_scope"] == "full_complex_antigen"
    assert full.metadata["vhh_environment_scope"] == "full_complex_vhh"
    assert truncated.metadata["antigen_environment_scope"] == "graph_antigen_nodes"
    assert truncated.metadata["vhh_environment_scope"] == "graph_vhh_nodes"
    assert not np.allclose(full.metadata["raw_antigen_energy"], truncated.metadata["raw_antigen_energy"])
    far_vhh = data.num_nodes - 1
    assert far_vhh not in sub.original_node_index.tolist()
    assert far_vhh in sub.vhh_context_index.tolist()
    assert not np.allclose(full.metadata["raw_vhh_environment_energy"],
                           truncated.metadata["raw_vhh_environment_energy"])


def test_antigen_neighbour_prefilter_is_exact(monkeypatch):
    import torch
    from nanoqc.model.model_egnn_pruning import build_ablation_subgraph
    data, _ = _graph_with_antigen_outside_radius()
    sub = build_ablation_subgraph(data, torch.arange(5), 6.0)
    filtered = _coarse_builder().build(sub).metadata
    monkeypatch.setattr(qubo, "_MAX_PSEUDO_ATOM_OFFSET", 1.0e6)
    unfiltered = _coarse_builder().build(sub).metadata
    # Same atom pairs are scored; only float summation order differs.
    for key in ("raw_antigen_energy", "raw_vhh_environment_energy"):
        assert np.allclose(filtered[key], unfiltered[key], rtol=0.0, atol=1e-12)


# ------------------------------------------------ chain breaks / graph version
def test_peptide_bond_detection_distinguishes_chain_breaks():
    previous = dict(c=(2.0, 1.2, 0.0))
    assert dataset_builder._peptide_bonded(previous, dict(n=(3.33, 1.2, 0.0)))
    assert not dataset_builder._peptide_bonded(previous, dict(n=(8.0, 1.2, 0.0)))


def test_allatom_phi_psi_fail_closed_across_chain_break():
    import pytest
    def residue(x0, n_offset=1.33):
        return {"name": "SER", "atoms": {
            "N": np.array([x0, 0.0, 0.0]), "CA": np.array([x0 + 1.46, 0.4, 0.0]),
            "C": np.array([x0 + 2.0, -0.6, 0.3])}}
    bonded = {"H:1": residue(0.0), "H:2": residue(3.33), "H:3": residue(6.66)}
    phi, psi = qubo._backbone_phi_psi(bonded, "H:2")
    assert np.isfinite([phi, psi]).all()
    broken = {"H:1": residue(0.0), "H:2": residue(3.33), "H:5": residue(20.0)}
    with pytest.raises(ValueError, match="chain break"):
        qubo._backbone_phi_psi(broken, "H:2")


def test_orchestrator_requires_the_current_graph_version():
    assert full.REQUIRED_GRAPH_VERSION == dataset_builder.VERSION


# ------------------------------------- biological assembly / antigen chains
_ASSEMBLY_PDB = """REMARK 350 BIOMOLECULE: 1
REMARK 350 SOFTWARE DETERMINED QUATERNARY STRUCTURE: MONOMERIC
REMARK 350 APPLY THE FOLLOWING TO CHAINS: C
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350 BIOMOLECULE: 2
REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: TRIMERIC
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000       10.00000
REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000 -1.000000        0.00000
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  ALA B   1       4.000   2.000   3.000  1.00  0.00           C
ATOM      3  CA  ALA C   1       7.000   2.000   3.000  1.00  0.00           C
END
"""


def test_biological_assembly_prefers_author_unit_and_names_symmetry_copies():
    import gemmi
    import nanoqc.data.audit_all_datasets as audit
    st = audit.biological_assembly_structure(gemmi.read_pdb_string(_ASSEMBLY_PDB))
    assert dict(st.info)[audit.STRUCTURE_SOURCE_KEY] == "biological_assembly:author_determined:2"
    positions = {chain.name: tuple(round(v, 3) for v in (chain[0][0].pos.x, chain[0][0].pos.y, chain[0][0].pos.z))
                 for chain in st[0]}
    assert positions == {"A": (1.0, 2.0, 3.0), "B": (4.0, 2.0, 3.0),
                         "A-2": (9.0, 2.0, -3.0), "B-2": (6.0, 2.0, -3.0)}
    # The same choice survives an mmCIF round trip (generators use subchains there).
    source = gemmi.read_pdb_string(_ASSEMBLY_PDB)
    source.setup_entities()
    from_cif = gemmi.make_structure_from_block(source.make_mmcif_document().sole_block())
    rebuilt = audit.biological_assembly_structure(from_cif)
    assert sorted(chain.name for chain in rebuilt[0]) == ["A", "A-2", "B", "B-2"]


def test_structures_without_assembly_annotation_fail_closed():
    import gemmi
    import pytest
    import nanoqc.data.audit_all_datasets as audit
    bare = "\n".join(line for line in _ASSEMBLY_PDB.splitlines() if not line.startswith("REMARK 350"))
    with pytest.raises(ValueError, match="no biological assembly annotation"):
        audit.biological_assembly_structure(gemmi.read_pdb_string(bare + "\n"))


def test_only_raw_pdb_subsets_are_rebuilt_from_the_assembly(tmp_path):
    import nanoqc.data.audit_all_datasets as audit
    path = tmp_path / "entry.pdb"
    path.write_text(_ASSEMBLY_PDB)
    for subset, expected in (("train_rcsb", "biological_assembly:author_determined:2"),
                             ("sabdab_vhh", "biological_assembly:author_determined:2"),
                             ("snac_db", "as_deposited_file")):
        st, _ = audit.read_structure(dict(path=str(path), member="", subset=subset, id="entry"))
        assert dict(st.info)[audit.STRUCTURE_SOURCE_KEY] == expected


def _chain(name, group, x0, sequence="QVQLV"):
    nodes = [dict(aa=aa, pos=(x0 + 3.8 * i, 0.0, 0.0), cb=(x0 + 3.8 * i, 1.5, 0.0))
             for i, aa in enumerate(sequence)]
    return dict(name=name, group=group, nodes=nodes)


def test_antigen_chains_are_the_ones_contacting_the_paratope():
    anchor = _chain("H", 0, 0.0)
    near = _chain("A", 1, 0.0, "KVFGR")
    near["nodes"] = [dict(n, pos=(n["pos"][0], 0.0, 6.0), cb=(n["cb"][0], 1.5, 6.0)) for n in near["nodes"]]
    far = _chain("C", 1, 0.0, "MKTAY")
    far["nodes"] = [dict(n, pos=(n["pos"][0], 0.0, 30.0), cb=(n["cb"][0], 1.5, 30.0)) for n in far["nodes"]]
    vhh_copy = _chain("H-2", 1, 0.0)
    vhh_copy["nodes"] = [dict(n, pos=(n["pos"][0], 0.0, 5.0), cb=(n["cb"][0], 1.5, 5.0)) for n in vhh_copy["nodes"]]
    row = dict(subset="sabdab_vhh", pdb_id="0XXX")
    kept, meta = dataset_builder.select_contacting_partner_chains(
        [anchor, near, far, vhh_copy], row, dict(structure_source="x"))
    assert [c["name"] for c in kept] == ["H", "A"]
    reasons = {d["chain"]: d["reason"] for d in meta["dropped_chains"]}
    assert reasons == {"C": "no_paratope_contact_within_7.5A", "H-2": "copy_of_vhh"}
    assert meta["antigen_contact_basis"] == "whole_vhh_chain_cdr_unmapped"
    # train_rcsb partners may be homomers of the anchor: identical sequence is kept.
    kept_rcsb, _ = dataset_builder.select_contacting_partner_chains(
        [anchor, near, vhh_copy], dict(subset="train_rcsb", pdb_id="0XXX"), dict(structure_source="x"))
    assert [c["name"] for c in kept_rcsb] == ["H", "A", "H-2"]
    import pytest
    with pytest.raises(ValueError, match="no partner chain"):
        dataset_builder.select_contacting_partner_chains([anchor, far], row, dict(structure_source="x"))


def test_all_atom_consumers_rebuild_exactly_the_graph_complex(tmp_path):
    import gemmi
    from types import SimpleNamespace
    import nanoqc.data.audit_all_datasets as audit
    path = tmp_path / "entry.pdb"
    path.write_text(_ASSEMBLY_PDB)
    graph = SimpleNamespace(structure_source="biological_assembly:author_determined:2", chain_ids=["A", "B-2"])
    out = audit.materialize_graph_complex(path, graph, tmp_path / "complex.cif")
    assert sorted(chain.name for chain in gemmi.read_structure(str(out))[0]) == ["A", "B-2"]
    unchanged = SimpleNamespace(structure_source="as_deposited_file", chain_ids=["A", "B", "C"])
    assert audit.materialize_graph_complex(path, unchanged, tmp_path / "same.cif") == path
    import pytest
    with pytest.raises(ValueError, match="differs from graph provenance"):
        audit.materialize_graph_complex(
            path, SimpleNamespace(structure_source="biological_assembly:author_determined:1", chain_ids=["A"]),
            tmp_path / "bad.cif")
