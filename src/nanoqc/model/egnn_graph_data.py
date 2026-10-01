"""Graph dataset, interface labels, seeding and DataLoader construction for EGNN training.

Split out of train_egnn_pruning.py.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm



class InterfaceGraphDataset(Dataset):
    """RAM-resident graph dataset with labels prepared during preload."""

    def __init__(self, data_list: Sequence[Data]) -> None:
        super().__init__(root=None)
        self.data_list = list(data_list)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, index: int) -> Data:
        return self.data_list[index]


def graph_protocol(data: Data) -> Dict[str, Any]:
    """Return the versioned scientific graph protocol encoded in one PyG graph."""

    required = (
        "graph_version", "edge_policy", "label_policy",
        "intra_chain_ca_cutoff_angstrom", "cross_partner_knn_k",
        "interface_label_cutoff_angstrom", "interface_sensitivity_cutoffs_angstrom",
        "min_interface_residues",
    )
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        raise ValueError(f"Graph lacks protocol metadata {missing}; rebuild with dataset version >=1.8")
    protocol = {
        "graph_version": str(data.graph_version),
        "edge_policy": str(data.edge_policy),
        "label_policy": str(data.label_policy),
        "intra_chain_ca_cutoff_angstrom": float(data.intra_chain_ca_cutoff_angstrom),
        "cross_partner_knn_k": int(data.cross_partner_knn_k),
        "interface_label_cutoff_angstrom": float(data.interface_label_cutoff_angstrom),
        "interface_sensitivity_cutoffs_angstrom": [
            float(v) for v in data.interface_sensitivity_cutoffs_angstrom
        ],
        "min_interface_residues": int(data.min_interface_residues),
    }
    if protocol["edge_policy"] != "intra_chain_ca_radius_plus_cross_partner_knn":
        raise ValueError("Unexpected graph edge policy; rebuild with current dataset builder")
    if protocol["label_policy"] != "cross_partner_heavy_atom_cutoff":
        raise ValueError("Unexpected graph label policy; rebuild with current dataset builder")
    if (protocol["intra_chain_ca_cutoff_angstrom"] <= 0
            or protocol["cross_partner_knn_k"] <= 0
            or protocol["interface_label_cutoff_angstrom"] <= 0
            or protocol["min_interface_residues"] <= 0):
        raise ValueError(f"Invalid graph protocol values: {protocol}")
    return protocol


def preload_graphs(paths: Sequence[Path], *, show_progress: bool) -> List[Data]:
    """Load every graph once, require one protocol, and attach labels."""

    iterator: Iterable[Path] = paths
    if show_progress:
        iterator = tqdm(paths, desc="Preloading graphs into RAM", unit="graph", dynamic_ncols=True)
    data_list = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in iterator
    ]
    expected_protocol: Optional[Dict[str, Any]] = None
    for data in data_list:
        protocol = graph_protocol(data)
        if expected_protocol is None:
            expected_protocol = protocol
        elif protocol != expected_protocol:
            raise ValueError(
                f"Mixed graph protocols in one training run: {expected_protocol} vs {protocol}"
            )
        data.y = interface_labels(data)
    return data_list


def interface_labels(data: Data) -> Tensor:
    """Return independently constructed heavy-atom interface labels.

    The exact heavy-atom cutoff is read from graph protocol metadata. Labels
    are never reconstructed from CA graph edges.
    """

    if not hasattr(data, "x") or not hasattr(data, "edge_index"):
        raise ValueError("Graph must contain x and edge_index")
    if data.x.ndim != 2 or data.x.size(1) != 21:
        raise ValueError(f"Expected x=[N,21], got {tuple(data.x.shape)}")
    if not hasattr(data, "interface_label"):
        raise ValueError(
            "Graph is missing interface_label; rebuild graphs with "
            "build_final_pyg_dataset.py version >= 1.8"
        )
    labels = data.interface_label.detach().cpu().to(torch.float32)
    if labels.shape != (data.num_nodes,):
        raise ValueError(
            f"interface_label must have shape [N], got {tuple(labels.shape)}"
        )
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("interface_label must contain only binary 0/1 values")
    graph_protocol(data)
    return labels

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_loader_worker(worker_id: int) -> None:
    """Give every DataLoader process a deterministic independent RNG stream."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_loader(
    data_list: Sequence[Data],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    options: Dict[str, Any] = {
        "dataset": InterfaceGraphDataset(data_list),
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "generator": generator,
        "worker_init_fn": seed_loader_worker,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(
        **options,
    )
