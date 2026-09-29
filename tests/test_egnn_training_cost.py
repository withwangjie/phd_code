"""Per-epoch work that is constant for a whole run must not be redone (A37)."""
import inspect
import math

import pytest
import torch

from nanoqc.model import train_egnn_pruning as training
from nanoqc.model import egnn_training_loop as loop

# Fail if a deprecated AMP API returns; unrelated dependency warnings retain
# the repository's existing filter policy.
pytestmark=pytest.mark.filterwarnings(r"error:.*torch\.cuda\.amp.*:FutureWarning")


def test_split_identity_is_computed_once_and_payload_is_unchanged(tmp_path, monkeypatch):
    from torch_geometric.data import Data

    paths = []
    for index, (pdb, family) in enumerate((("1abc", "c1"), ("2def", "c2"), ("3ghi", "c1"))):
        graph = Data(pos=torch.zeros(2, 3), x=torch.zeros(2, 21), edge_index=torch.zeros(2, 0, dtype=torch.long))
        graph.pdb_id, graph.family_structure_cluster = pdb, family
        graph.graph_version, graph.interface_label_cutoff_angstrom = "1.11", 4.5
        graph.edge_policy = "intra_chain_ca_radius_plus_cross_partner_knn"
        graph.label_policy = "cross_partner_heavy_atom_cutoff"
        graph.interface_sensitivity_cutoffs_angstrom, graph.cross_partner_knn_k = [3.5, 5.0], 3
        graph.intra_chain_ca_cutoff_angstrom, graph.min_interface_residues = 8.0, 15
        path = tmp_path / f"g{index}.pt"
        torch.save(graph, path)
        paths.append(path)
    train_paths, validation_paths = paths[:2], paths[2:]

    loads = []
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (loads.append(a[0]), real_load(*a, **k))[1])
    identity = training.run_split_identity(train_paths, validation_paths)
    once = len(loads)
    assert once == len(paths) + 1, "identity reads each graph once plus the protocol graph"

    payload_kwargs = dict(
        model=torch.nn.Linear(1, 1), optimizer=torch.optim.AdamW(torch.nn.Linear(1, 1).parameters()),
        scaler=torch.amp.GradScaler("cuda", enabled=False), epoch=3,
        metrics=training.BinaryMetrics(loss=0.5, roc_auc=0.6, pr_auc=0.6, threshold=0.5, precision=0.5,
                                       recall=0.5, f1=0.5, default_precision=0.5, default_recall=0.5,
                                       default_f1=0.5, positives=10, negatives=90),
        train_loss=0.4, model_config={"hidden_dim": 32}, training_config={"max_epochs": 50},
        train_paths=train_paths, validation_paths=validation_paths, history=[], best_auc=0.6,
        best_epoch=3, epochs_without_improvement=0, train_positives=10, train_negatives=90)

    loads.clear()
    recomputing = training._checkpoint_payload(**payload_kwargs)
    assert len(loads) == once, "the old path re-read every graph"
    loads.clear()
    precomputed = training._checkpoint_payload(**payload_kwargs, split_identity=identity)
    assert loads == [], "a precomputed identity must not touch the disk"
    assert precomputed["split"] == recomputing["split"]
    assert precomputed["graph_protocol"] == recomputing["graph_protocol"]


def test_main_computes_the_split_identity_outside_the_epoch_loop():
    source = inspect.getsource(training.main)
    before, _, after = source.partition("for epoch in range(start_epoch")
    assert "run_split_identity(" in before, "identity must be computed before the loop"
    assert "run_split_identity(" not in after
    assert "split_identity=split_identity" in after


def test_training_step_synchronizes_once_and_keeps_the_guard():
    source = inspect.getsource(loop.train_one_epoch)
    assert source.count("float(loss.detach())") == 1
    assert "torch.isfinite(loss)" not in source
    assert "math.isfinite(loss_value)" in source
    # The guard still rejects both nan and inf, exactly as torch.isfinite did.
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert not math.isfinite(float(torch.tensor(bad, dtype=torch.float32)))
    assert math.isfinite(float(torch.tensor(0.25, dtype=torch.float32)))


def test_non_finite_loss_still_raises(monkeypatch):
    """The A20 FP16 guard fires on the first non-finite batch."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    class NanModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, pos, edge_index, return_logits=False):
            return torch.full((x.shape[0],), float("nan")) + self.weight

    graph = Data(x=torch.zeros(2, 21), pos=torch.zeros(2, 3),
                 edge_index=torch.zeros(2, 0, dtype=torch.long), y=torch.zeros(2))
    model = NanModel()
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        loop.train_one_epoch(
            model, DataLoader([graph], batch_size=1),
            torch.optim.AdamW(model.parameters()),
            torch.amp.GradScaler("cuda", enabled=False),
            device=torch.device("cpu"), pos_weight=torch.tensor(1.0), gradient_clip=5.0,
            epoch=1, amp_enabled=False, pin_memory=False, is_main_process=False)


def test_amp_disabled_training_matches_fp32_update_and_validation():
    """Modern AMP API with enabled=False must retain the actual FP32 update."""
    import copy
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear=torch.nn.Linear(1,1)
        def forward(self,x,pos,edge_index,return_logits=False):
            assert x.dtype==torch.float32
            output=self.linear(x).reshape(-1)
            assert output.dtype==torch.float32
            return output

    model=Model()
    with torch.no_grad():
        model.linear.weight.fill_(0.25);model.linear.bias.fill_(-0.1)
    reference=copy.deepcopy(model)
    graph=Data(x=torch.arange(4,dtype=torch.float32).reshape(-1,1),pos=torch.zeros(4,3),
               edge_index=torch.zeros(2,0,dtype=torch.long),y=torch.tensor([0.,1.,0.,1.]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=.01)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(
        reference(graph.x,graph.pos,graph.edge_index),graph.y,pos_weight=torch.tensor(1.))
    expected.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(),5.)
    reference_optimizer.step()
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    assert scaler.state_dict()=={} and not scaler.is_enabled()
    loss=loop.train_one_epoch(model,DataLoader([graph],batch_size=1),optimizer,scaler,
        device=torch.device("cpu"),pos_weight=torch.tensor(1.),gradient_clip=5.,epoch=1,
        amp_enabled=False,pin_memory=False,is_main_process=False)
    assert loss==float(expected.detach())
    for actual,wanted in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
    metrics=loop.validate(model,DataLoader([graph],batch_size=1),device=torch.device("cpu"),
        pos_weight=torch.tensor(1.),epoch=1,amp_enabled=False,pin_memory=False)
    assert math.isfinite(metrics.loss) and metrics.positives==2 and metrics.negatives==2
