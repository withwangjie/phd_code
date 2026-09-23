"""DB5.5 PDB reservation must not depend on graph construction success."""

from nanoqc.data.build_final_pyg_dataset import db55_reserved_pdb_ids


def test_db55_reserves_failed_and_ineligible_pairs():
    pairs = [
        {"id": "1abc", "valid": True},
        {"id": "2def", "valid": False},
        {"id": "3ghi", "valid": True},  # graph creation may still fail
    ]

    assert db55_reserved_pdb_ids(pairs) == {"1ABC", "2DEF", "3GHI"}
