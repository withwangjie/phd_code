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


def _graph(path: Path, vhh: str, antigen: str, cdr3: str, cluster: str, pdb: str,
           subset: str = "snac_db") -> Path:
    graph = Data(x=torch.zeros(1, 3))
    graph.vhh_sequences = [vhh]
    graph.antigen_sequences = [antigen]
    graph.cdr3_seq = cdr3
    graph.family_structure_cluster = cluster
    graph.subset_source = subset
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
        source = raw / f"{pdb}.pdb"
        source.write_text(f"REMARK {pdb}\nEND\n")
        source_id = f"snac_db/{pdb}.pdb"
        source_sha = sha256_file(source)
        path = _graph(train / f"snac_db__{pdb.upper()}.pt", _sequence(rng), _sequence(rng, 200),
                      _sequence(rng, 14), f"cluster_{index:06d}", pdb)
        manifest.append(dict(
            split="train", path=f"graphs/train/{path.name}", pdb_id=pdb.upper(),
            subset_source="snac_db", source_id=source_id,
            source_structure_sha256=source_sha, sha256=sha256_file(path),
            family_structure_cluster=f"cluster_{index:06d}"))
        audit_rows.append(dict(
            id=source_id, subset="snac_db", pdb_id=pdb.upper(), path=str(source),
            member="", valid=True, source_structure_sha256=source_sha))
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
    source = tmp_path / "raw" / "99zz.pdb"
    source.write_text("REMARK 99zz\nEND\n")
    source_id = "snac_db/99zz.pdb"
    source_sha = sha256_file(source)
    manifest.append(dict(
        split="train", path=f"graphs/train/{twin.name}", pdb_id="99ZZ",
        subset_source="snac_db", source_id=source_id,
        source_structure_sha256=source_sha, sha256=sha256_file(twin),
        family_structure_cluster="cluster_999999"))
    (dataset / "graph_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    details = audit / "data_audit_details.jsonl"
    details.write_text(details.read_text() + json.dumps(dict(
        id=source_id, subset="snac_db", pdb_id="99ZZ", path=str(source),
        member="", valid=True, source_structure_sha256=source_sha)) + "\n")

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


def test_unreadable_later_source_leaves_training_split_intact(tmp_path, monkeypatch):
    dataset, audit = _dataset(tmp_path)
    original_manifest = (dataset / "graph_manifest.json").read_bytes()
    original_graphs = {p.name for p in (dataset / "graphs" / "train").glob("*.pt")}
    real_copy = carve.copy_source_structure
    calls = 0

    def fail_second(source, pdb, destination, expected_sha256):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("unreadable later source")
        return real_copy(source, pdb, destination, expected_sha256)

    monkeypatch.setattr(carve, "copy_source_structure", fail_second)
    with pytest.raises(OSError, match="unreadable later source"):
        carve.main(["--dataset-dir", str(dataset), "--audit-dir", str(audit),
                    "--out-json", str(tmp_path / "holdout.json"), "--min-clusters", "2"])
    assert (dataset / "graph_manifest.json").read_bytes() == original_manifest
    assert {p.name for p in (dataset / "graphs" / "train").glob("*.pt")} == original_graphs
    assert not list((dataset / "graphs" / "holdout").glob("*.pt"))


def test_failed_holdout_metadata_write_rolls_back_moved_graphs(tmp_path, monkeypatch):
    dataset, audit = _dataset(tmp_path)
    original_manifest = (dataset / "graph_manifest.json").read_bytes()
    original_graphs = {p.name for p in (dataset / "graphs" / "train").glob("*.pt")}

    def fail_write(*_args):
        raise OSError("metadata write failed")

    monkeypatch.setattr(carve, "write_outputs", fail_write)
    with pytest.raises(OSError, match="metadata write failed"):
        carve.main(["--dataset-dir", str(dataset), "--audit-dir", str(audit),
                    "--out-json", str(tmp_path / "holdout.json"), "--min-clusters", "2"])
    assert (dataset / "graph_manifest.json").read_bytes() == original_manifest
    assert {p.name for p in (dataset / "graphs" / "train").glob("*.pt")} == original_graphs
    assert not list((dataset / "graphs" / "holdout").glob("*.pt"))


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


def test_foldseek_can_bind_the_pair_table_to_one_run_universe(tmp_path, monkeypatch):
    """A standalone audit and a run's own audit can disagree; --run-dir removes the gap."""
    import yaml
    from nanoqc.data import prepare_external_vhh as prep

    prep_dir = tmp_path / "prep"
    (prep_dir / "audit").mkdir(parents=True)
    (prep_dir / "study_pdb_ids.txt").write_text("1abc\n2def\n")
    run = tmp_path / "run"
    (run / "audit").mkdir(parents=True)
    (run / "audit" / "cluster_universe.txt").write_text("1abc\n2def\n9shv\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(dict(
        paths=dict(data_root=str(tmp_path / "data"), dataset_dir="dataset"),
        data_audit=dict(max_resolution_angstrom=3.0),
        queue_freeze=dict(homology_isolation={}, independence_clustering=dict(pair_tsv=str(tmp_path / "p.tsv"))),
        external_validation=dict(external_vhh=dict(graph_dir="", source_structure_dir="", min_clusters=10)))))

    seen = {}
    monkeypatch.setattr(prep, "run_module", lambda module, argv: seen.update(argv=argv))
    common = ["--config", str(config_path), "--prep-dir", str(prep_dir), "foldseek"]
    assert prep.main(common + ["--run-dir", str(run)]) == 0
    assert seen["argv"][seen["argv"].index("--universe") + 1] == str(run / "audit" / "cluster_universe.txt")
    assert seen["argv"][seen["argv"].index("--audit-dir") + 1] == str(run / "audit")

    with pytest.raises(SystemExit, match="has not reached queue_freeze"):
        prep.main(common + ["--run-dir", str(tmp_path / "nothing")])


def _merge(dataset: Path, count: int, cluster: str = "cluster_giant") -> set[str]:
    """Put the first `count` training graphs in one structure cluster: one layered component."""
    names = set()
    for path in sorted((dataset / "graphs" / "train").glob("*.pt"))[:count]:
        graph = torch.load(path, weights_only=False)
        graph.family_structure_cluster = cluster
        torch.save(graph, path)
        names.add(path.name)
    return names


def test_a_component_larger_than_one_fold_always_trains(tmp_path, monkeypatch):
    """A11: a giant component is never the internal validation fold nor the holdout."""
    monkeypatch.setattr(training, "PIN_MIN_COMPONENT", 2)
    for fold in range(1, training.SPLIT_FOLDS):
        root = tmp_path / f"fold{fold}"
        dataset, audit = _dataset(root)
        giant = _merge(dataset, 12)  # 12 * 5 > 40
        payload = _carve(dataset, audit, root / "holdout.json", fold=fold, min_clusters=1)
        assert payload["pinned_to_training"]["component_sizes"] == [12]
        assert not giant & {path.name for path in (dataset / "graphs" / "holdout").glob("*.pt")}
        train, validation = training.split_paths(sorted((dataset / "graphs" / "train").glob("*.pt")), 0)
        assert giant <= {path.name for path in train} and not giant & {path.name for path in validation}


def test_a_component_at_the_pin_boundary_fails_the_carve(tmp_path, monkeypatch):
    """Not pinned in the full pool (8 * 5 == 40) but pinned in the smaller remaining one."""
    monkeypatch.setattr(training, "PIN_MIN_COMPONENT", 2)
    dataset, audit = _dataset(tmp_path)
    giant = _merge(dataset, 8)
    paths = sorted((dataset / "graphs" / "train").glob("*.pt"))
    component = next(c for c in training.layered_components(paths) if {p.name for p in c} == giant)
    fold = next(f for f in range(1, training.SPLIT_FOLDS) if f != training.component_fold(component)
                and carve.select_components(training.layered_components(paths), [f], len(paths)))
    with pytest.raises(SystemExit, match="pin boundary"):
        _carve(dataset, audit, tmp_path / "holdout.json", fold=fold, min_clusters=1)


def test_several_folds_are_held_out_together_by_rule_not_by_search():
    """A13: the lowest non-validation folds, never the fullest one."""
    assert carve.holdout_folds(5, 1, 0) == [1]
    assert carve.holdout_folds(5, 2, 0) == [1, 2]
    assert carve.parse_folds("2,1") == [1, 2]


def test_holding_out_two_folds_moves_both_and_records_them(tmp_path):
    dataset, audit = _dataset(tmp_path)
    paths = sorted((dataset / "graphs" / "train").glob("*.pt"))
    components = training.layered_components(paths)
    expected = {p.name for c in components if training.component_fold(c) in (1, 2) for p in c}
    payload = _carve(dataset, audit, tmp_path / "holdout.json", fold="1,2", min_clusters=1)
    assert payload["folds_held_out"] == [1, 2] and payload["fold"] == [1, 2]
    assert {p.name for p in (dataset / "graphs" / "holdout").glob("*.pt")} == expected
    assert not expected & {p.name for p in (dataset / "graphs" / "train").glob("*.pt")}


def test_the_holdout_cannot_swallow_every_non_validation_fold(tmp_path):
    dataset, audit = _dataset(tmp_path)
    with pytest.raises(SystemExit) as raised:  # argparse error: message goes to stderr
        _carve(dataset, audit, tmp_path / "holdout.json", fold="1,2,3,4", min_clusters=1)
    assert raised.value.code == 2


def test_sabdab_in_selected_primary_component_is_quarantined_not_scored(tmp_path):
    dataset, audit = _dataset(tmp_path)
    train_dir = dataset / "graphs" / "train"
    primary = sorted(train_dir.glob("*.pt"))[0]
    primary_graph = torch.load(primary, map_location="cpu", weights_only=False)

    # Choose a deterministic auxiliary filename whose two-member component is
    # not the reserved internal-validation fold.
    aux_path = None
    for suffix in range(100):
        candidate = train_dir / f"sabdab_vhh__AUX{suffix:02d}.pt"
        if training.component_fold([primary, candidate]) != training.VALIDATION_FOLD:
            aux_path = candidate
            break
    assert aux_path is not None
    aux_pdb = "9aux"
    _graph(aux_path, primary_graph.vhh_sequences[0], _sequence(random.Random(101), 200),
           _sequence(random.Random(102), 14), "cluster_aux", aux_pdb, subset="sabdab_vhh")

    source = tmp_path / "raw" / f"{aux_pdb}.pdb"
    source.write_text("REMARK auxiliary\nEND\n")
    source_id = f"sabdab_vhh/{aux_pdb}.pdb"
    source_sha = sha256_file(source)
    manifest_path = dataset / "graph_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.append(dict(
        split="train", path=f"graphs/train/{aux_path.name}", pdb_id=aux_pdb.upper(),
        subset_source="sabdab_vhh", source_id=source_id,
        source_structure_sha256=source_sha, sha256=sha256_file(aux_path),
        family_structure_cluster="cluster_aux"))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    details = audit / "data_audit_details.jsonl"
    details.write_text(details.read_text() + json.dumps(dict(
        id=source_id, subset="sabdab_vhh", pdb_id=aux_pdb.upper(), path=str(source),
        member="", valid=True, source_structure_sha256=source_sha)) + "\n")

    component = next(c for c in training.layered_components(sorted(train_dir.glob("*.pt")))
                     if aux_path in c)
    fold = training.component_fold(component)
    assert fold != training.VALIDATION_FOLD
    payload = _carve(dataset, audit, tmp_path / "holdout_aux.json",
                     fold=fold, min_clusters=1)

    held = {p.name for p in (dataset / "graphs" / "holdout").glob("*.pt")}
    quarantined = {p.name for p in (dataset / "graphs" / "holdout_quarantine").glob("*.pt")}
    assert primary.name in held
    assert aux_path.name in quarantined
    assert aux_path.name not in held
    assert all(t["subset_source"] == "snac_db" for t in payload["targets"])
    assert any(q["subset_source"] == "sabdab_vhh"
               and q["reason"] == "auxiliary_sabdab_component_member"
               for q in payload["quarantined"])
    assert aux_path.name not in {p.name for p in train_dir.glob("*.pt")}


def test_holdout_uses_exact_source_id_not_first_row_for_same_pdb(tmp_path):
    dataset, audit = _dataset(tmp_path)
    train_dir = dataset / "graphs" / "train"
    paths = sorted(train_dir.glob("*.pt"))
    primary = next(path for path in paths
                   if training.component_fold([path]) != training.VALIDATION_FOLD)
    manifest = json.loads((dataset / "graph_manifest.json").read_text())
    row = next(r for r in manifest if r["path"] == f"graphs/train/{primary.name}")
    pdb = row["pdb_id"].lower()

    # Prepend a lower-priority same-PDB source whose bytes differ. A PDB-only
    # lookup would pick this wrong structure; exact source_id binding must not.
    bogus = tmp_path / "raw" / f"{pdb}_bogus.pdb"
    bogus.write_text("BOGUS\nEND\n")
    bogus_row = dict(
        id=f"sabdab_vhh/bogus_{pdb}.pdb", subset="sabdab_vhh",
        pdb_id=pdb.upper(), path=str(bogus), member="", valid=True,
        source_structure_sha256=sha256_file(bogus),
    )
    details = audit / "data_audit_details.jsonl"
    details.write_text(json.dumps(bogus_row) + "\n" + details.read_text())

    fold = training.component_fold([primary])
    payload = _carve(dataset, audit, tmp_path / "holdout_exact.json",
                     fold=fold, min_clusters=1)
    target = next(t for t in payload["targets"] if t["pdb_id"] == pdb)
    assert target["source_id"] == row["source_id"]
    binding = payload["source_structures"][pdb]
    assert binding["source_id"] == row["source_id"]
    copied = Path(binding["path"])
    assert copied.read_text() != bogus.read_text()
