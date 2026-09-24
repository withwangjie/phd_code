"""Antigen-fold holdout carved from the training split (PROTOCOL_AMENDMENTS.md A10)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from nanoqc.common.repo_io import sha256_file
from nanoqc.data import carve_holdout_clusters as carve
from nanoqc.model import train_egnn_pruning as training

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def _sequence(rng: random.Random, length: int = 120) -> str:
    return "".join(rng.choice(RESIDUES) for _ in range(length))


def _graph(path: Path, vhh: str, antigen: str, cdr3: str, cluster: str, pdb: str) -> Path:
    graph = Data(x=torch.zeros(1, 3))
    graph.vhh_sequences = [vhh]
    graph.antigen_sequences = [antigen]
    graph.cdr3_seq = cdr3
    graph.family_structure_cluster = cluster
    graph.subset_source = "snac_db"
    torch.save(graph, path)
    return path


def _dataset(tmp_path: Path, count: int = 40) -> tuple[Path, Path]:
    """A training split of `count` mutually non-homologous complexes, plus its audit."""
    rng = random.Random(11)
    dataset = tmp_path / "dataset"
    train = dataset / "graphs" / "train"
    train.mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest, audit_rows = [], []
    for index in range(count):
        pdb = f"{index + 10:02d}ab"
        path = _graph(train / f"snac_db__{pdb.upper()}.pt", _sequence(rng), _sequence(rng, 200),
                      _sequence(rng, 14), f"cluster_{index:06d}", pdb)
        manifest.append(dict(split="train", path=f"graphs/train/{path.name}", pdb_id=pdb.upper(),
                             sha256=sha256_file(path), family_structure_cluster=f"cluster_{index:06d}"))
        source = raw / f"{pdb}.pdb"
        source.write_text(f"REMARK {pdb}\nEND\n")
        audit_rows.append(dict(pdb_id=pdb.upper(), path=str(source), member="", valid=True))
    (dataset / "graph_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "data_audit_details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in audit_rows))
    return dataset, audit


def _carve(dataset: Path, audit: Path, out: Path, **over) -> dict:
    argv = ["--dataset-dir", str(dataset), "--audit-dir", str(audit), "--out-json", str(out)]
    for key, value in {"min_clusters": 2, **over}.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    assert carve.main(argv) == 0
    return json.loads(out.read_text())


def test_whole_components_move_and_training_can_no_longer_see_them(tmp_path):
    dataset, audit = _dataset(tmp_path)
    payload = _carve(dataset, audit, tmp_path / "holdout.json")

    train = sorted(p.name for p in (dataset / "graphs" / "train").glob("*.pt"))
    held = sorted(p.name for p in (dataset / "graphs" / "holdout").glob("*.pt"))
    assert held and train and not set(train) & set(held)
    assert len(held) == payload["graphs"] == len(payload["targets"])
    assert payload["holdout_components"] >= 2 and payload["fold"] == 1
    assert payload["internal_validation_fold"] == training.VALIDATION_FOLD

    # The manifest follows the files, with fresh hashes and the new split.
    manifest = json.loads((dataset / "graph_manifest.json").read_text())
    rows = {row["path"]: row for row in manifest}
    for target in payload["targets"]:
        row = rows[target["path"]]
        assert row["split"] == "holdout"
        assert row["sha256"] == sha256_file(dataset / target["path"])
    assert sum(row["split"] == "train" for row in manifest) == len(train)

    # One audited raw structure per holdout PDB, so the independence audit can bind them.
    structures = dataset / "holdout_source_structures"
    assert {p.name for p in structures.glob("*")} == {f"{t['pdb_id']}.pdb" for t in payload["targets"]}
    assert payload["source_structure_dir"] == str(structures.resolve())


def test_the_holdout_never_splits_a_homologous_component(tmp_path):
    """Two complexes sharing a VHH must land on the same side, whatever their folds."""
    dataset, audit = _dataset(tmp_path, count=40)
    train = dataset / "graphs" / "train"
    original = sorted(train.glob("*.pt"))[0]
    shared = torch.load(original, map_location="cpu", weights_only=False)
    twin = train / "snac_db__99ZZ.pt"
    _graph(twin, shared.vhh_sequences[0], _sequence(random.Random(5), 200),
           _sequence(random.Random(6), 14), "cluster_999999", "99zz")
    manifest = json.loads((dataset / "graph_manifest.json").read_text())
    manifest.append(dict(split="train", path=f"graphs/train/{twin.name}", pdb_id="99ZZ",
                         sha256=sha256_file(twin), family_structure_cluster="cluster_999999"))
    (dataset / "graph_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (tmp_path / "raw" / "99zz.pdb").write_text("REMARK 99zz\nEND\n")
    details = audit / "data_audit_details.jsonl"
    details.write_text(details.read_text() + json.dumps(
        dict(pdb_id="99ZZ", path=str(tmp_path / "raw" / "99zz.pdb"), member="", valid=True)) + "\n")

    _carve(dataset, audit, tmp_path / "holdout.json")
    held = {path.name for path in (dataset / "graphs" / "holdout").glob("*.pt")}
    kept = {path.name for path in train.glob("*.pt")}
    pair = {twin.name, original.name}
    assert pair <= held or pair <= kept, f"homologous pair split: held={held & pair}, kept={kept & pair}"


def test_too_few_holdout_components_fails_before_any_training(tmp_path):
    dataset, audit = _dataset(tmp_path, count=8)
    with pytest.raises(SystemExit, match="below the preregistered minimum 50"):
        _carve(dataset, audit, tmp_path / "holdout.json", min_clusters=50)
    assert not (dataset / "graphs" / "holdout").exists() or not list(
        (dataset / "graphs" / "holdout").glob("*.pt"))


def test_the_carve_is_deterministic_and_runs_once(tmp_path):
    first = _carve(*_dataset(tmp_path / "a"), tmp_path / "a.json")
    second = _carve(*_dataset(tmp_path / "b"), tmp_path / "b.json")
    assert [t["pdb_id"] for t in first["targets"]] == [t["pdb_id"] for t in second["targets"]]

    dataset, audit = _dataset(tmp_path / "c")
    _carve(dataset, audit, tmp_path / "c.json")
    with pytest.raises(SystemExit, match="carved once per run"):
        _carve(dataset, audit, tmp_path / "c2.json")


def test_the_holdout_fold_cannot_be_the_internal_validation_fold(tmp_path):
    dataset, audit = _dataset(tmp_path)
    with pytest.raises(SystemExit):
        _carve(dataset, audit, tmp_path / "holdout.json", fold=training.VALIDATION_FOLD)


def test_removing_holdout_components_leaves_the_internal_split_unchanged(tmp_path):
    """Whole components leave; every remaining component keeps its fold."""
    dataset, audit = _dataset(tmp_path)
    train = dataset / "graphs" / "train"
    before = {path.name: training.component_fold(component)
              for component in training.layered_components(sorted(train.glob("*.pt")))
              for path in component}
    expected_validation = sorted(name for name, fold in before.items() if fold == training.VALIDATION_FOLD)
    _carve(dataset, audit, tmp_path / "holdout.json")
    after = {path.name: training.component_fold(component)
             for component in training.layered_components(sorted(train.glob("*.pt")))
             for path in component}
    assert all(after[name] == before[name] for name in after)
    assert sorted(name for name, fold in after.items() if fold == training.VALIDATION_FOLD) == expected_validation


def test_foldseek_step_clusters_the_study_pdbs_alone_without_pass1(tmp_path, monkeypatch):
    """With the holdout there are no external candidates, so the universe is study_pdb_ids.txt."""
    import yaml
    from nanoqc.data import prepare_external_vhh as prep

    prep_dir = tmp_path / "prep"
    (prep_dir / "audit").mkdir(parents=True)
    (prep_dir / "study_pdb_ids.txt").write_text("1abc\n2def\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(dict(
        paths=dict(data_root=str(tmp_path / "data"), dataset_dir="dataset"),
        data_audit=dict(max_resolution_angstrom=3.0),
        queue_freeze=dict(homology_isolation={}, independence_clustering=dict(pair_tsv=str(tmp_path / "p.tsv"))),
        external_validation=dict(external_vhh=dict(graph_dir="", source_structure_dir="", min_clusters=10)))))

    seen = {}
    monkeypatch.setattr(prep, "run_module", lambda module, argv: seen.update(module=module, argv=argv))
    assert prep.main(["--config", str(config_path), "--prep-dir", str(prep_dir), "foldseek"]) == 0
    assert seen["module"] == "nanoqc.data.build_foldseek_pairs"
    assert "--universe" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--universe") + 1] == str(prep_dir / "study_pdb_ids.txt")
    assert "--external-candidates" not in seen["argv"]  # nothing external to add

    # Stale pass-1 artifacts must not widen the universe while the holdout is in use.
    (prep_dir / "foldseek_universe.txt").write_text("1abc\n2def\n9zzz\n")
    (prep_dir / "selection_pass1").mkdir()
    (prep_dir / "selection_pass1" / "candidates.json").write_text("{}")
    prep.main(["--config", str(config_path), "--prep-dir", str(prep_dir), "foldseek"])
    assert seen["argv"][seen["argv"].index("--universe") + 1] == str(prep_dir / "study_pdb_ids.txt")
    assert "--external-candidates" not in seen["argv"]

    # A configured external set does use pass 1's wider universe.
    external = yaml.safe_load(config_path.read_text())
    external["external_validation"]["external_vhh"].update(
        graph_dir=str(tmp_path / "ext" / "graphs"), source_structure_dir=str(tmp_path / "ext" / "structures"))
    config_path.write_text(yaml.safe_dump(external))
    prep.main(["--config", str(config_path), "--prep-dir", str(prep_dir), "foldseek"])
    assert seen["argv"][seen["argv"].index("--universe") + 1] == str(prep_dir / "foldseek_universe.txt")
    assert "--external-candidates" in seen["argv"]


def test_cluster_adequacy_counts_the_holdout_before_any_training(tmp_path):
    from nanoqc.pipeline import run_full_experiment as full

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = ([dict(split="test_snac_hard", pdb_id=f"h{i:03d}") for i in range(12)]
                + [dict(split="holdout", pdb_id=f"o{i:03d}") for i in range(4)]
                + [dict(split="train", pdb_id="t001")])
    (dataset / "graph_manifest.json").write_text(json.dumps(manifest))
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps([f"h{i:03d}" for i in range(12)]))
    cluster_map = tmp_path / "clusters.json"
    cluster_map.write_text(json.dumps({row["pdb_id"]: f"c_{row['pdb_id']}" for row in manifest}))
    config = dict(statistics=dict(min_qc_clusters=10, min_scaling_clusters=10,
                                  min_primary_clusters=10, min_rq5_clusters=10),
                  external_validation=dict(external_vhh=dict(required=True, min_clusters=10)))

    report = full.cluster_adequacy(config, dataset, selected, cluster_map)
    assert report["holdout_pdbs"] == 4 and report["holdout_clusters"] == 4
    assert not report["adequate"]
    assert any("antigen-fold holdout" in shortfall for shortfall in report["shortfalls"])

    # Enough holdout clusters: the gate passes and still reports the count.
    manifest += [dict(split="holdout", pdb_id=f"o{i:03d}") for i in range(4, 10)]
    (dataset / "graph_manifest.json").write_text(json.dumps(manifest))
    cluster_map.write_text(json.dumps({row["pdb_id"]: f"c_{row['pdb_id']}" for row in manifest}))
    ok = full.cluster_adequacy(config, dataset, selected, cluster_map)
    assert ok["adequate"] and ok["holdout_clusters"] == 10


def test_smoke_check_reads_this_run_dataset_and_skips_before_queue_freeze():
    """The recovery sub-check must not fall back to the pilot's standalone default path."""
    import inspect
    from nanoqc.pipeline import run_full_experiment as full

    source = inspect.getsource(full.Orchestrator.stage_smoke_check)
    assert '"--dataset", str(self.dataset_dir())' in source
    assert '"--data-root"' in source
    # Skipped, not failed, while the dataset does not exist yet.
    assert 'if smoke_manifest.is_file():' in source
    assert "No smoke check could run yet" in source
    # The fallback that caused the failure was the pilot's standalone default.
    from nanoqc.experiments import run_real_complex_pilot as pilot
    assert "dataset_clean_500" in inspect.getsource(pilot.main)  # still the standalone default
    assert "dataset_clean_500" not in source  # but never what the orchestrator relies on
