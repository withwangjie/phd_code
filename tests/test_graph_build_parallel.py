import ast
import inspect
from types import SimpleNamespace

import nanoqc.data.build_final_pyg_dataset as build
from nanoqc.data import graph_build_parallel as parallel


def test_worker_settings_are_exactly_what_main_rebinds():
    tree = ast.parse(inspect.getsource(build.main))
    rebound = {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names}
    assert set(parallel.SETTINGS) == rebound - {"PEAK_RSS"}


def test_sequences_copies_only_the_homology_attributes():
    graph = SimpleNamespace(vhh_sequences=["QVQ"], antigen_sequences=["MKV"], cdr3_seq="CAA",
                            subset_source="snac_db", pos=object(), edge_index=object())
    light = parallel.sequences(graph)
    assert set(vars(light)) == set(parallel.HOMOLOGY_FIELDS)
    assert build.layered_graph_homology(light, light) == build.layered_graph_homology(graph, graph)
    # A graph lacking an attribute keeps lacking it, so getattr defaults still apply.
    assert not hasattr(parallel.sequences(SimpleNamespace(cdr3_seq="CAA")), "vhh_sequences")


def _records():
    vhh = ["QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKG",
           "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG",
           "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKG"]
    antigen = ["MKVLAAGIVGLLLAQAAPAHAEETLKQ", "MKVLAAGIVGLLLAQAAPAHAEETLKR", "GSHMTTPSLPEQWRLAAEKLNEY"]
    cdr = ["CAADRSTWYGYDY", "CAKDRSYGMDV", "CAADRSTWYGYDF"]
    subsets = ["snac_db", "sabdab_vhh", "train_rcsb"]
    return [SimpleNamespace(vhh_sequences=[v], antigen_sequences=[a], cdr3_seq=c, subset_source=s)
            for v, a, c, s in zip(vhh, antigen, cdr, subsets)]


def test_parallel_homology_rows_match_the_sequential_values():
    records = _records()
    with parallel.pool(2, reference=records) as executor:
        lower = list(executor.map(parallel.reference_row, range(len(records))))
        full = list(executor.map(parallel.homology_row, records))
    for i, row in enumerate(lower):
        assert row == [build.layered_graph_homology(records[i], records[j]) for j in range(i)]
    for record, row in zip(records, full):
        assert row == [build.layered_graph_homology(record, other) for other in records]


def test_save_graph_task_returns_the_unchanged_result_and_a_peak_rss(monkeypatch):
    monkeypatch.setattr(build, "save_graph", lambda *a, **k: ({"args": a, "kwargs": k}, None))
    record, error, peak = parallel.save_graph_task(("row", "train", "out"), {"family_structure_cluster": "c1"})
    assert record == {"args": ("row", "train", "out"), "kwargs": {"family_structure_cluster": "c1"}}
    assert error is None and isinstance(peak, int)
