"""CDR-priority baseline must use only uniquely mapped observed loop residues."""

from types import SimpleNamespace

import pytest
import torch

from nanoqc.model.model_egnn_pruning import AA, _ablation_cdr_indices


def graph(vhh: str, cdr: str, antigen: str = "ACDEFG") -> SimpleNamespace:
    residues = vhh + antigen
    x = torch.zeros((len(residues), 21), dtype=torch.float32)
    for index, aa in enumerate(residues):
        x[index, AA.index(aa)] = 1
    x[len(vhh):, -1] = 1
    return SimpleNamespace(
        x=x, cdr3_seq=cdr,
        node_chain_id=torch.tensor([0] * len(vhh) + [1] * len(antigen)),
    )


def test_cdr_h3_maps_two_unobserved_internal_residues() -> None:
    annotated = "ATDPECYRVRGYYNGEYDY"
    observed = "ATDPECYRVRGYYNGDY"
    vhh = "QVQLVESGGGSVQAGGSLRLSCAAS" + "C" + observed + "WGQGTQVTVSS"
    data = graph(vhh, annotated)
    start = vhh.index(observed)
    assert _ablation_cdr_indices(data) == list(range(start, start + len(observed)))


def test_cdr_h3_exact_match_still_maps() -> None:
    cdr = "ATDPECYRVRGYYNGEYDY"
    vhh = "QVQLVESGGGSVQAGGSLRLSCAAS" + "C" + cdr + "WGQGTQVTVSS"
    data = graph(vhh, cdr)
    start = vhh.index(cdr)
    assert _ablation_cdr_indices(data) == list(range(start, start + len(cdr)))


def test_partial_cdr_mapping_requires_unique_conserved_flanks() -> None:
    annotated = "ATDPECYRVRGYYNGEYDY"
    observed = "ATDPECYRVRGYYNGDY"
    for vhh in ("A" + observed + "WGQGT", "C" + observed + "AGQGT",
                "C" + observed + "WGQGTC" + observed + "WGQGT"):
        with pytest.raises(ValueError, match="absent/ambiguous"):
            _ablation_cdr_indices(graph(vhh, annotated))
