"""Regression tests for five formal-run blockers found in the reorg audit."""
from __future__ import annotations

import inspect
import json
import sys
from unittest.mock import patch

import pytest

import nanoqc.data.build_final_pyg_dataset as dataset_builder
import nanoqc.experiments.batch_benchmark_hard_set as bbh
import nanoqc.pipeline.run_full_experiment as full
from nanoqc.data.build_independence_cluster_map import main as cluster_main, norm_id


# 1. Formal graphs get their family/structure cluster before validate_graph().
def test_make_graph_sets_family_cluster_before_validation():
    source = inspect.getsource(dataset_builder.make_graph)
    assigned = source.index("graph.family_structure_cluster=")
    validated = source.index("validate_graph(graph)")
    assert assigned < validated
    assert "family_structure_cluster=family_structure_cluster" in inspect.getsource(
        dataset_builder.save_graph)


# 2. Time-mode statistics are named "<baseline>_time" by the paired-statistics driver.
def test_primary_effect_name_matches_paired_statistics_naming():
    assert full.primary_qc_effect_name("sa", "outputs") == "sa"
    assert full.primary_qc_effect_name("sa", "time") == "sa_time"
    assert 'name=baseline if args.budget_mode=="outputs" else baseline+"_time"' in inspect.getsource(
        bbh._paired_statistics_main)


# 3. Exact-oracle timing lives on each metrics row, not at the top of a case file.
def test_rotamer_resolution_reads_oracle_time_from_metrics_rows():
    source = inspect.getsource(full.Orchestrator.stage_method_sensitivity)
    assert 'case["oracle_seconds"]' not in source
    assert 'case.get("metrics")' in source
    writer = inspect.getsource(bbh._ablation_run_case)
    assert "oracle_seconds=oracle_seconds" in writer


# 5. Foldseek entry names with chain suffixes map to the PDB ID; junk still fails.
def test_norm_id_accepts_foldseek_entry_names():
    for name, expected in (
        ("1abc", "1abc"), ("1ABC", "1abc"), ("1abc.cif.gz", "1abc"),
        ("1abc.cif.gz_A", "1abc"), ("7rdr_B", "7rdr"), ("/x/1abc.pdb_AB", "1abc"),
        ("1abc_assembly1", "1abc"), ("1abc_assembly1.cif.gz_A", "1abc"),
    ):
        assert norm_id(name) == expected, name


def test_norm_id_rejects_non_pdb_names():
    for name in ("x", "1abc_extra", "t37_model.pdb", "12345", "1abc_extra_A", ""):
        with pytest.raises(ValueError):
            norm_id(name)


def test_cluster_map_accepts_chain_suffixed_foldseek_table(tmp_path):
    pairs = tmp_path / "pairs.tsv"
    out = tmp_path / "map.json"
    pairs.write_text(
        "query\ttarget\tqtmscore\n1abc.cif.gz_A\t2def.cif.gz_B\t0.8\n2def.cif.gz_B\t3ghi_A\t0.2\n",
        encoding="utf-8")
    universe = tmp_path / "universe.txt"
    universe.write_text("1abc\n2def\n3ghi\n", encoding="utf-8")
    argv = ["cluster-map", "--pairs", str(pairs), "--out-json", str(out),
            "--min-score", "0.5", "--score-semantics", "qtmscore", "--universe", str(universe)]
    with patch.object(sys, "argv", argv):
        assert cluster_main() == 0
    mapping = json.loads(out.read_text(encoding="utf-8"))
    assert mapping["1abc"] == mapping["2def"] != mapping["3ghi"]


def test_cluster_universe_keeps_only_pdb_ids():
    source = inspect.getsource(full.Orchestrator.stage_queue_freeze)
    assert 're.fullmatch(r"[a-z0-9]{4}",pdb)' in source
