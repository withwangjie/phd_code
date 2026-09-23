from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from nanoqc.data.audit_external_vhh_independence import graph_sequences
from nanoqc.data.safe_graph_load import load_graph


def test_external_graph_restricted_roundtrip(tmp_path) -> None:
    path=tmp_path/"graph.pt"
    graph=Data(x=torch.ones((2,3)))
    graph.pdb_id="1ABC"
    graph.graph_version="1.6"
    graph.vhh_sequences=["ACD"]
    graph.antigen_sequences=["EFG"]
    graph.cdr3_seq="ACD"
    torch.save(graph,path)
    assert load_graph(path).pdb_id=="1ABC"
    assert graph_sequences(path)["pdb_id"]=="1abc"


def test_external_graph_rejects_pickle_global(tmp_path) -> None:
    path=tmp_path/"malicious.pt"
    marker=tmp_path/"executed.txt"

    class Unsafe:
        def __reduce__(self):
            return (eval,(f"open({str(marker)!r}, 'w').write('executed')",))

    torch.save(Unsafe(),path)
    with pytest.raises(Exception):
        load_graph(path)
    assert not marker.exists()
