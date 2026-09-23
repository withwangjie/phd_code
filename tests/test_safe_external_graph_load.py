from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from nanoqc.data.audit_external_vhh_independence import graph_sequences
from nanoqc.data.safe_graph_load import load_graph


def test_external_graph_restricted_roundtrip(tmp_path) -> None:
    path=tmp_path/"graph.pt"
    order="ACDEFGHIKLMNPQRSTVWY"
    features=torch.zeros((6,21))
    for index,aa in enumerate("ACDEFG"):
        features[index,order.index(aa)]=1
        features[index,20]=0 if index<3 else 1
    graph=Data(x=features,node_chain_id=torch.tensor([0,0,0,1,1,1]))
    graph.pdb_id="1ABC"
    graph.graph_version="1.6"
    graph.vhh_sequences=["ACD"]
    graph.antigen_sequences=["EFG"]
    graph.chain_sequences=["ACD","EFG"]
    graph.chain_groups=[0,1]
    graph.cdr3_seq="ACD"
    graph.cdr3_len=3
    torch.save(graph,path)
    assert load_graph(path).pdb_id=="1ABC"
    assert graph_sequences(path)["pdb_id"]=="1abc"


def test_external_graph_rejects_forged_sequence_metadata(tmp_path) -> None:
    path=tmp_path/"graph.pt"
    graph=Data(x=torch.zeros((2,21)),node_chain_id=torch.tensor([0,1]))
    graph.x[0,0]=1;graph.x[1,1]=1;graph.x[1,20]=1
    graph.pdb_id="1ABC"
    graph.chain_sequences=["A","C"]
    graph.chain_groups=[0,1]
    graph.vhh_sequences=["W"]
    graph.antigen_sequences=["C"]
    graph.cdr3_seq="A";graph.cdr3_len=1
    torch.save(graph,path)
    with pytest.raises(ValueError,match="partner sequences disagree"):
        graph_sequences(path)


def test_unannotated_training_cdr_is_allowed_but_external_cdr_is_required(tmp_path) -> None:
    path=tmp_path/"graph.pt"
    features=torch.zeros((2,21))
    features[0,0]=1;features[1,1]=1;features[1,20]=1
    graph=Data(x=features,node_chain_id=torch.tensor([0,1]))
    graph.pdb_id="1ABC"
    graph.chain_sequences=["A","C"];graph.chain_groups=[0,1]
    graph.vhh_sequences=["A"];graph.antigen_sequences=["C"]
    graph.cdr3_seq="";graph.cdr3_len=0
    torch.save(graph,path)
    assert graph_sequences(path)["cdr_h3"]==""
    with pytest.raises(ValueError,match="CDR-H3 sequence"):
        graph_sequences(path,tmp_path)


def test_external_graph_is_bound_to_raw_structure(tmp_path) -> None:
    source_dir=tmp_path/"structures"
    source_dir.mkdir()
    (source_dir/"1abc.pdb").write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n"
        "ATOM      2  CA  CYS B   1       4.000   5.000   6.000  1.00 20.00           C\nEND\n",
        encoding="utf-8")
    features=torch.zeros((2,21))
    features[0,0]=1;features[1,1]=1;features[1,20]=1
    graph=Data(x=features,node_chain_id=torch.tensor([0,1]),
               pos=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]]))
    graph.pdb_id="1ABC";graph.graph_version="1.6"
    graph.chain_sequences=["A","C"];graph.chain_groups=[0,1]
    graph.vhh_sequences=["A"];graph.antigen_sequences=["C"]
    graph.cdr3_seq="A";graph.cdr3_len=1
    graph.residue_ids=["A:1","B:1"]
    path=tmp_path/"graph.pt"
    torch.save(graph,path)
    record=graph_sequences(path,source_dir)
    assert record["source_structure_sha256"]
    graph.pos[0,0]=2.0
    torch.save(graph,path)
    with pytest.raises(ValueError,match="CA coordinate disagrees"):
        graph_sequences(path,source_dir)


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


def test_training_cdr_with_unmodeled_residues_does_not_abort_audit(tmp_path) -> None:
    # Annotated CDR-H3 "AXW" is not a substring of the modeled VHH chain "AC":
    # acceptable for a hash-bound training graph, rejected for an external one.
    path=tmp_path/"graph.pt"
    features=torch.zeros((2,21))
    features[0,0]=1;features[1,1]=1;features[1,20]=1
    graph=Data(x=features,node_chain_id=torch.tensor([0,1]))
    graph.pdb_id="1ABC"
    graph.chain_sequences=["A","C"];graph.chain_groups=[0,1]
    graph.vhh_sequences=["A"];graph.antigen_sequences=["C"]
    graph.cdr3_seq="AXW";graph.cdr3_len=3
    torch.save(graph,path)
    assert graph_sequences(path)["cdr_h3"]=="AXW"
    with pytest.raises(ValueError,match="CDR-H3 sequence"):
        graph_sequences(path,tmp_path)
