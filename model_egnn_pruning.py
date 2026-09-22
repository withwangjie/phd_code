"""SE(3)-equivariant interface scoring and spatial pruning for protein graphs.

The module expects PyG ``Data`` objects with:

* ``x``: ``[N, 21]`` scalar node features. The final column is the binary
  interaction-partner label (VHH=0, partner=1).
* ``pos``: ``[N, 3]`` Cartesian coordinates.
* ``edge_index``: ``[2, E]`` directed COO edges. Undirected graphs should
  store both directions, as produced by ``build_final_pyg_dataset.py``.

No orientation-dependent vector is passed to a scalar MLP. Edge messages use
only scalar node features and squared Euclidean distances, while coordinate
updates are scalar multiples of relative displacement vectors. Consequently,
the network is equivariant to rotations and translations (and invariant at
the scalar interface-score output).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Data


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Construct a two-layer SiLU MLP used by the EGNN blocks."""

    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class E_GCL(nn.Module):
    """Equivariant graph convolution layer for scalar features and 3D points.

    ``edge_index[0]`` stores source nodes ``j`` and ``edge_index[1]`` stores
    target nodes ``i``. For every directed edge ``j -> i`` the layer computes

    ``m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, edge_attr_ij)``.

    Messages are summed at target node ``i``. Coordinates and scalar features
    are then updated as

    ``x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij)`` and
    ``h_i' = LayerNorm(h_i + phi_h(h_i, sum_j m_ij))``.

    Args:
        hidden_dim: Width of node features and edge messages.
        edge_attr_dim: Number of optional scalar edge features.
        mlp_hidden_dim: Internal MLP width. Defaults to ``hidden_dim``.
        coord_scale: Bounds the initial magnitude of each learned coordinate
            coefficient after the final ``tanh``.
        dropout: Dropout probability in the scalar node update.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int = 0,
        mlp_hidden_dim: Optional[int] = None,
        coord_scale: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if edge_attr_dim < 0:
            raise ValueError("edge_attr_dim cannot be negative")
        if coord_scale <= 0:
            raise ValueError("coord_scale must be positive")

        inner_dim = mlp_hidden_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.coord_scale = coord_scale

        self.edge_mlp = _mlp(
            2 * hidden_dim + 1 + edge_attr_dim,
            inner_dim,
            hidden_dim,
        )
        self.coord_mlp = _mlp(
            hidden_dim,
            inner_dim,
            1,
            final_activation=nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

        # A zero initial coordinate update makes early optimization stable.
        # The weights remain trainable, so the layer learns non-zero motion.
        coordinate_output = self.coord_mlp[-2]
        if not isinstance(coordinate_output, nn.Linear):
            raise TypeError("Unexpected coordinate MLP layout")
        nn.init.zeros_(coordinate_output.weight)
        nn.init.zeros_(coordinate_output.bias)

    def _validate_inputs(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
    ) -> None:
        """Validate shapes early so malformed graph batches fail clearly."""

        if h.ndim != 2 or h.size(-1) != self.hidden_dim:
            raise ValueError(
                f"h must have shape [N, {self.hidden_dim}], got {tuple(h.shape)}"
            )
        if pos.ndim != 2 or pos.shape != (h.size(0), 3):
            raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= h.size(0)
        ):
            raise ValueError("edge_index contains an out-of-range node index")
        if edge_attr is not None:
            expected = (edge_index.size(1), self.edge_attr_dim)
            if tuple(edge_attr.shape) != expected:
                raise ValueError(
                    f"edge_attr must have shape {expected}, got {tuple(edge_attr.shape)}"
                )

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply one equivariant message-passing and residual update step."""

        self._validate_inputs(h, pos, edge_index, edge_attr)
        source, target = edge_index

        relative = pos[target] - pos[source]  # x_i - x_j
        squared_distance = relative.square().sum(dim=-1, keepdim=True)

        edge_inputs = [h[target], h[source], squared_distance]
        if self.edge_attr_dim:
            if edge_attr is None:
                edge_attr = h.new_zeros((edge_index.size(1), self.edge_attr_dim))
            edge_inputs.append(edge_attr)
        messages = self.edge_mlp(torch.cat(edge_inputs, dim=-1))

        aggregated_messages = h.new_zeros((h.size(0), self.hidden_dim))
        aggregated_messages.index_add_(0, target, messages.to(dtype=aggregated_messages.dtype))

        coordinate_coefficients = self.coord_scale * self.coord_mlp(messages)
        coordinate_messages = relative * coordinate_coefficients
        coordinate_delta = pos.new_zeros(pos.shape)
        coordinate_delta.index_add_(0, target, coordinate_messages.to(dtype=coordinate_delta.dtype))
        updated_pos = pos + coordinate_delta

        node_delta = self.node_mlp(torch.cat([h, aggregated_messages], dim=-1))
        updated_h = self.node_norm(h + node_delta)
        return updated_h, updated_pos


class EGNNInterfaceScorer(nn.Module):
    """Four-layer EGNN that assigns an interface probability to every residue.

    Args:
        input_dim: Input scalar feature count. The delivered dataset uses 21.
        hidden_dim: Hidden scalar width.
        num_layers: Number of E_GCL layers; defaults to the requested four.
        edge_attr_dim: Optional scalar edge-feature count.
        dropout: Dropout probability used by node updates and the output head.
        coord_scale: Per-edge coordinate coefficient scale in each E_GCL.
    """

    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 128,
        num_layers: int = 4,
        edge_attr_dim: int = 0,
        dropout: float = 0.1,
        coord_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_dim = input_dim
        self.edge_attr_dim = edge_attr_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            E_GCL(
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                coord_scale=coord_scale,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(
        self,
        node_features: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return learned scalar embeddings and equivariantly updated points."""

        if node_features.ndim != 2 or node_features.size(-1) != self.input_dim:
            raise ValueError(
                f"node_features must have shape [N, {self.input_dim}], "
                f"got {tuple(node_features.shape)}"
            )
        h = self.input_projection(node_features)
        coordinates = pos
        for layer in self.layers:
            h, coordinates = layer(h, coordinates, edge_index, edge_attr)
        return h, coordinates

    def forward_logits(
        self,
        node_features: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        *,
        return_coordinates: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return node logits, optionally with the final equivariant coordinates.

        Use this method with ``BCEWithLogitsLoss`` during training for better
        numerical stability than applying binary cross entropy to probabilities.
        """

        h, updated_pos = self.encode(node_features, pos, edge_index, edge_attr)
        logits = self.output_head(h).squeeze(-1)
        if return_coordinates:
            return logits, updated_pos
        return logits

    def forward(
        self,
        node_features: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        *,
        return_coordinates: bool = False,
        return_logits: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return probabilities by default, or raw logits for DDP training."""

        result = self.forward_logits(
            node_features,
            pos,
            edge_index,
            edge_attr,
            return_coordinates=return_coordinates,
        )
        if return_coordinates:
            logits, updated_pos = result
            return (logits if return_logits else torch.sigmoid(logits)), updated_pos
        return result if return_logits else torch.sigmoid(result)


def extract_top_interface_subgraph(
    data: Data,
    model: EGNNInterfaceScorer,
    probability_threshold: float = 0.5,
    min_active: int = 5,
    max_active: int = 15,
    environment_radius: float = 6.0,
    antigen_guidance_weight: float = 0.25,
    antigen_proximity_scale: float = 6.0,
) -> Data:
    """Adaptively select active VHH residues and their frozen microenvironment.

    VHH/group-0 residues are ranked by EGNN probability plus a label-free
    nearest-antigen proximity prior. Fewer than ``min_active`` candidates triggers
    a top-score fallback; more than ``max_active`` candidates are truncated by
    the composite score. Every non-active
    residue whose CA lies within ``environment_radius`` Angstrom (inclusive) of
    an active residue is retained as a frozen environment node.

    Added output attributes:

    * ``interface_score``: model probability for each retained node.
    * ``is_active``: marks residues whose rotamers may vary.
    * ``is_frozen_environment``: marks retained fixed background residues.
    * ``selected_vhh_mask``: compatibility alias of ``is_active``.
    * ``original_node_index``: maps subgraph nodes back to input node indices.
    * adaptive selection parameters and the threshold-selected count.

    Args:
        data: One complex graph. Batched graphs are intentionally rejected.
        model: Trained interface scorer.
        probability_threshold: Active-residue probability cutoff.
        min_active: Minimum active count, filled by top scores if necessary.
        max_active: Maximum active count, truncated by top scores if necessary.
        environment_radius: Inclusive CA cutoff for frozen surroundings.
        antigen_guidance_weight: Weight in [0,1] for nearest-antigen proximity.
        antigen_proximity_scale: Positive exponential decay length (Angstrom) for antigen proximity.

    Returns:
        A node-induced ``Data`` containing disjoint active and frozen masks.
    """

    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1]")
    if not 1 <= min_active <= max_active:
        raise ValueError("Require 1 <= min_active <= max_active")
    if max_active > 15:
        raise ValueError("max_active cannot exceed 15 under the 30-bit budget")
    if environment_radius <= 0:
        raise ValueError("environment_radius must be positive")
    if not 0.0 <= antigen_guidance_weight <= 1.0:
        raise ValueError("antigen_guidance_weight must be in [0, 1]")
    if not math.isfinite(antigen_proximity_scale) or antigen_proximity_scale <= 0:
        raise ValueError("antigen_proximity_scale must be positive and finite")
    if not hasattr(data, "x") or not hasattr(data, "pos") or not hasattr(data, "edge_index"):
        raise ValueError("data must contain x, pos, and edge_index")
    if data.x.ndim != 2 or data.x.size(-1) != model.input_dim:
        raise ValueError(
            f"data.x must have shape [N, {model.input_dim}], got {tuple(data.x.shape)}"
        )
    if data.pos.shape != (data.num_nodes, 3):
        raise ValueError("data.pos must have shape [N, 3]")
    if getattr(data, "batch", None) is not None:
        batch = data.batch
        if batch.numel() and int(batch.max()) > 0:
            raise ValueError("Pass one complex Data object, not a multi-graph Batch")

    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model has no parameters") from exc

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            edge_attr = getattr(data, "edge_attr", None)
            probabilities = model(
                data.x.to(model_device),
                data.pos.to(model_device),
                data.edge_index.to(model_device),
                None if edge_attr is None else edge_attr.to(model_device),
            )
    finally:
        model.train(was_training)

    probabilities = probabilities.detach().to(data.x.device)
    group = data.x[:, -1]
    zero = torch.zeros((), dtype=group.dtype, device=group.device)
    vhh_mask = torch.isclose(group, zero)
    if not torch.all(vhh_mask | torch.isclose(group, torch.ones_like(group))):
        raise ValueError("data.x[:, -1] must contain only binary 0/1 partner labels")

    vhh_indices = torch.nonzero(vhh_mask, as_tuple=False).flatten()
    antigen_indices = torch.nonzero(~vhh_mask, as_tuple=False).flatten()
    if vhh_indices.numel() == 0:
        raise ValueError("graph contains no VHH/group-0 nodes")
    if antigen_indices.numel() == 0:
        raise ValueError("graph contains no antigen/group-1 nodes")
    available = int(vhh_indices.numel())
    if available < min_active:
        raise ValueError(
            f"graph contains only {available} VHH nodes; cannot satisfy Top-{min_active} fallback"
        )
    required_minimum = min_active
    permitted_maximum = min(max_active, available)
    vhh_probabilities = probabilities[vhh_indices]
    nearest_antigen = torch.cdist(
        data.pos[vhh_indices], data.pos[antigen_indices]
    ).min(dim=1).values
    antigen_proximity = torch.exp(-nearest_antigen / float(antigen_proximity_scale))
    vhh_scores = (
        (1.0 - antigen_guidance_weight) * vhh_probabilities
        + antigen_guidance_weight * antigen_proximity
    )
    threshold_local = torch.nonzero(
        vhh_scores >= probability_threshold, as_tuple=False
    ).flatten()
    threshold_count = int(threshold_local.numel())
    if threshold_count < required_minimum:
        selected_local = torch.topk(vhh_scores, required_minimum).indices
    elif threshold_count > permitted_maximum:
        threshold_scores = vhh_scores[threshold_local]
        selected_local = threshold_local[
            torch.topk(threshold_scores, permitted_maximum).indices
        ]
    else:
        selected_local = threshold_local
    active_indices = vhh_indices[selected_local]

    all_indices = torch.arange(data.num_nodes, device=data.pos.device)
    distances = torch.cdist(data.pos[active_indices], data.pos)
    within_environment = (distances <= environment_radius).any(dim=0)
    active_global_mask = torch.zeros(
        data.num_nodes, dtype=torch.bool, device=data.pos.device
    )
    active_global_mask[active_indices] = True
    frozen_global_mask = within_environment & ~active_global_mask
    frozen_indices = all_indices[frozen_global_mask]
    kept_indices = torch.cat([active_indices, frozen_indices]).unique(sorted=True)
    subgraph = data.subgraph(kept_indices)
    subgraph.original_node_index = kept_indices
    composite_scores = probabilities.clone()
    composite_scores[vhh_indices] = vhh_scores
    subgraph.egnn_probability = probabilities[kept_indices]
    subgraph.interface_score = composite_scores[kept_indices]
    subgraph.is_active = torch.isin(kept_indices, active_indices)
    subgraph.is_frozen_environment = torch.isin(kept_indices, frozen_indices)
    if torch.any(subgraph.is_active & subgraph.is_frozen_environment):
        raise AssertionError("Active and frozen masks must be disjoint")
    if not torch.all(subgraph.is_active | subgraph.is_frozen_environment):
        raise AssertionError("Every retained node must be active or frozen")
    subgraph.selected_vhh_mask = subgraph.is_active.clone()
    subgraph.pruning_probability_threshold = float(probability_threshold)
    subgraph.pruning_min_active = int(min_active)
    subgraph.pruning_max_active = int(max_active)
    subgraph.pruning_environment_radius = float(environment_radius)
    subgraph.pruning_antigen_guidance_weight = float(antigen_guidance_weight)
    subgraph.pruning_antigen_proximity_scale = float(antigen_proximity_scale)
    subgraph.threshold_candidate_count = threshold_count
    subgraph.active_residue_count = int(subgraph.is_active.sum())
    subgraph.frozen_residue_count = int(subgraph.is_frozen_environment.sum())
    return subgraph


def _random_radius_graph(num_nodes: int, cutoff: float = 2.5) -> Data:
    """Create one deterministic toy complex for executable smoke tests."""

    generator = torch.Generator().manual_seed(7)
    pos = torch.randn((num_nodes, 3), generator=generator)
    distances = torch.cdist(pos, pos)
    adjacency = (distances < cutoff) & (distances > 0)
    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous().long()

    residue_types = torch.randint(0, 20, (num_nodes,), generator=generator)
    x = torch.zeros((num_nodes, 21), dtype=torch.float32)
    x[torch.arange(num_nodes), residue_types] = 1.0
    x[num_nodes // 2 :, -1] = 1.0
    return Data(x=x, pos=pos.float(), edge_index=edge_index)




# Fixed-budget experimental controls; adaptive production API remains available.
import numpy as np
AA = "ACDEFGHIKLMNPQRSTVWY"

def _ablation_cdr_indices(data: Any) -> list[int]:
    """Map annotated CDR sequence exactly; never assume numbering ranges."""
    seq = getattr(data, "cdr3_seq", "")
    if not seq:
        raise ValueError("CDR-priority ablation requires annotated cdr3_seq.")
    matches = []
    for chain in torch.unique(data.node_chain_id).tolist():
        ids = torch.where((data.node_chain_id == chain) & (data.x[:, -1] == 0))[0].tolist()
        sequence = "".join(AA[int(data.x[i, :20].argmax())] for i in ids)
        for start in range(len(sequence)):
            if sequence.startswith(seq, start):
                matches.append(ids[start:start+len(seq)])
    if len(matches) != 1:
        raise ValueError("CDR sequence is absent/ambiguous in observed residues.")
    return matches[0]


def select_ablation_active(data: Any, method: str, k: int, seed: int,
                  scorer: EGNNInterfaceScorer | None,
                  antigen_guidance_weight: float = 0.25,
                  antigen_proximity_scale: float = 6.0,
                  contact_ca_cutoff: float = 8.0) -> torch.Tensor:
    """Select exactly k VHH sites without using solver outcomes or native labels."""
    if not 0.0 <= antigen_guidance_weight <= 1.0:
        raise ValueError("antigen_guidance_weight must be in [0,1]")
    if not math.isfinite(antigen_proximity_scale) or antigen_proximity_scale <= 0:
        raise ValueError("antigen_proximity_scale must be positive finite")
    if not math.isfinite(contact_ca_cutoff) or contact_ca_cutoff <= 0:
        raise ValueError("contact_ca_cutoff must be positive finite")
    candidates = torch.where(data.x[:, -1] == 0)[0]
    if len(candidates) < k:
        raise ValueError("Not enough VHH residues for matched site budget.")
    partner = torch.where(data.x[:, -1] == 1)[0]
    if not len(partner):
        raise ValueError("No antigen/group-1 residues.")
    distances = torch.cdist(data.pos[candidates], data.pos[partner])
    nearest = distances.min(1).values
    scores = torch.zeros(data.num_nodes)
    if method == "egnn":
        if scorer is None:
            raise ValueError("Trained checkpoint required for EGNN ablation.")
        with torch.no_grad():
            model_scores = scorer(data.x, data.pos, data.edge_index).cpu()
        proximity = torch.exp(-nearest / float(antigen_proximity_scale))
        scores[candidates] = (
            (1.0-antigen_guidance_weight)*model_scores[candidates]
            + antigen_guidance_weight*proximity
        )
    elif method == "distance":
        scores[candidates] = -nearest
    elif method == "contact":
        # Geometry baseline only: CA neighborhood count, never native interface_label.
        scores[candidates] = (distances < float(contact_ca_cutoff)).sum(1).float()
    elif method == "random":
        ids = np.random.default_rng(seed).choice(candidates.numpy(), k, replace=False)
        return torch.tensor(sorted(ids), dtype=torch.long)
    elif method == "cdr":
        # Prioritize observed CDR-H3, then contact degree within each tier.
        scores[_ablation_cdr_indices(data)] += float(scores.max()) + 1
    elif method != "contact":
        raise ValueError(method)
    order = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), i))
    return torch.tensor(sorted(order[:k]), dtype=torch.long)


def build_ablation_subgraph(data: Any, active: torch.Tensor, radius: float) -> Any:
    """Keep a fixed Active set while changing frozen environment radius."""
    distance = torch.cdist(data.pos, data.pos[active]).min(1).values
    keep = torch.where(distance <= radius)[0]
    sub = data.subgraph(keep)
    sub.original_node_index = keep
    sub.is_active = torch.isin(keep, active)
    sub.selected_vhh_mask = sub.is_active.clone()
    sub.is_frozen_environment = ~sub.is_active
    # PyG does not necessarily slice Python metadata lists.
    sub.residue_ids = [data.residue_ids[i] for i in keep.tolist()]
    return sub


if __name__ == "__main__":
    torch.manual_seed(7)
    toy = _random_radius_graph(num_nodes=40)
    scorer = EGNNInterfaceScorer(
        input_dim=21,
        hidden_dim=64,
        num_layers=4,
        dropout=0.0,
    ).eval()
    # Exercise a genuinely non-zero coordinate update in the equivariance test.
    with torch.no_grad():
        for egnn_layer in scorer.layers:
            coordinate_output = egnn_layer.coord_mlp[-2]
            assert isinstance(coordinate_output, nn.Linear)
            nn.init.normal_(coordinate_output.weight, mean=0.0, std=1e-3)

    probabilities, updated_pos = scorer(
        toy.x,
        toy.pos,
        toy.edge_index,
        return_coordinates=True,
    )
    assert probabilities.shape == (toy.num_nodes,)
    assert updated_pos.shape == toy.pos.shape
    assert torch.all((probabilities >= 0) & (probabilities <= 1))

    # Scalar scores must be invariant and updated coordinates equivariant under
    # a proper rotation followed by translation.
    random_matrix = torch.randn((3, 3))
    rotation, _ = torch.linalg.qr(random_matrix)
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = torch.tensor([2.0, -3.0, 1.5])
    transformed_pos = toy.pos @ rotation + translation
    transformed_probabilities, transformed_updated_pos = scorer(
        toy.x,
        transformed_pos,
        toy.edge_index,
        return_coordinates=True,
    )
    assert torch.allclose(probabilities, transformed_probabilities, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        updated_pos @ rotation + translation,
        transformed_updated_pos,
        atol=2e-5,
        rtol=2e-5,
    )

    pruned = extract_top_interface_subgraph(
        toy,
        scorer,
        probability_threshold=1.0,
        min_active=5,
        max_active=15,
        environment_radius=100.0,
    )
    assert pruned.x.shape[1] == 21
    assert int(pruned.is_active.sum()) == 5
    assert int(pruned.is_frozen_environment.sum()) == pruned.num_nodes - 5
    assert not torch.any(pruned.is_active & pruned.is_frozen_environment)
    assert pruned.interface_score.shape == (pruned.num_nodes,)
    assert pruned.edge_index.shape[0] == 2
    print(
        "EGNN smoke test passed: "
        f"{toy.num_nodes} input nodes -> {pruned.num_nodes} pruned nodes."
    )
