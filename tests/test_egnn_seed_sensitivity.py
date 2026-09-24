"""End-to-end check of the development-only EGNN seed-sensitivity analysis."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from nanoqc.model import egnn_seed_sensitivity as sensitivity
from nanoqc.model.model_egnn_pruning import EGNNInterfaceScorer

AA = "ACDEFGHIKLMNPQRSTVWY"


def _graph(path: Path, rng: random.Random, family: str) -> Path:
    vhh, antigen = 10, 6
    n = vhh + antigen
    x = torch.zeros((n, 21))
    for i in range(n):
        x[i, AA.index(rng.choice("DEKRQNSTYFILMHW"))] = 1.0
    x[vhh:, -1] = 1.0
    pos = torch.tensor([[3.8 * i, 0.0, 0.0] for i in range(vhh)]
                       + [[3.0 * i, 5.0 + rng.random(), 0.0] for i in range(antigen)], dtype=torch.float32)
    distances = torch.cdist(pos, pos)
    graph = Data(x=x, pos=pos, edge_index=((distances < 8.0) & (distances > 0)).nonzero().t().long())
    residues = lambda k: "".join(rng.choice(AA) for _ in range(k))
    graph.vhh_sequences = [residues(120)]
    graph.antigen_sequences = [residues(200)]
    graph.cdr3_seq = residues(12)
    graph.family_structure_cluster = family
    graph.subset_source = "sabdab_vhh"
    torch.save(graph, path)
    return path


def _checkpoint(path: Path, seed: int, auc: float) -> Path:
    torch.manual_seed(seed)
    model = EGNNInterfaceScorer()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_config={}, model_state_dict=model.state_dict(), graph_protocol={},
                    early_stopping=dict(best_auc=auc)), path)
    return path


def test_seed_sensitivity_reports_auc_spread_and_site_stability(tmp_path: Path):
    rng = random.Random(0)
    data_dir = tmp_path / "train"
    data_dir.mkdir()
    for index in range(15):
        _graph(data_dir / f"g{index:02d}.pt", rng, f"fam{index}")
    primary = _checkpoint(tmp_path / "primary" / "best.pt", 1, 0.80)
    replicates = [_checkpoint(tmp_path / f"r{i}" / "best.pt", 10 + i, 0.78 + 0.01 * i) for i in (1, 2)]
    out_json, out_md = tmp_path / "summary.json", tmp_path / "summary.md"
    assert sensitivity.main([
        "--data-dir", str(data_dir), "--primary-checkpoint", str(primary),
        "--replicate-checkpoints", *map(str, replicates), "--active-sites", "3",
        "--out-json", str(out_json), "--out-md", str(out_md),
    ]) == 0
    payload = json.loads(out_json.read_text())
    assert payload["models"] == 3
    assert payload["roc_auc_range"] == pytest.approx([0.79, 0.80])
    assert payload["graphs_compared"] >= 1
    assert 0.0 <= payload["site_jaccard_vs_primary_min"] <= payload["site_jaccard_vs_primary_mean"] <= 1.0
    assert "never used to choose a model" in out_md.read_text()


def test_jaccard_edge_cases():
    assert sensitivity.jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert sensitivity.jaccard([1, 2], [3, 4]) == 0.0
    assert sensitivity.jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
