"""Restricted loading for PyG graph files, including third-party graphs."""
from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage


def load_graph(path: str | Path) -> Data:
    # These are the only non-tensor classes in this project's PyG Data format.
    # An unknown pickle global fails instead of being imported or executed.
    with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        graph = torch.load(path, map_location="cpu", weights_only=True)
    if type(graph) is not Data:
        raise ValueError(f"Expected a PyG Data graph: {path}")
    return graph
