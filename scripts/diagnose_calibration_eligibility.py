#!/usr/bin/env python3
"""Inspect calibration exclusions without changing a formal run's artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from nanoqc.data.safe_graph_load import load_graph
from nanoqc.model.model_egnn_pruning import AA, load_interface_scorer, select_ablation_active
from nanoqc.qubo.subgraph_to_qubo import (
    PyRosettaRotamerProvider,
    _THREE_LETTER,
    _dunbrack_templates_for_site,
    _nearest_dunbrack_bin,
    chi1_well_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    run = args.run_dir.resolve(strict=True)
    config = yaml.safe_load((run / "frozen_config.yaml").read_text(encoding="utf-8"))
    qc = config["qc_benchmark"]
    cal = qc["energy_calibration"]
    rot = qc["rotamer_model"]
    provenance = json.loads((run / cal["training_csv"]).with_suffix(".provenance.json").read_text())
    dataset = run / config["paths"].get("dataset_dir", "dataset")
    manifest = json.loads((dataset / "graph_manifest.json").read_text())
    train = {str(row["pdb_id"]).lower(): row for row in manifest if row.get("split") == "train"}
    checkpoint = run / config["paths"].get("checkpoint_dir", "checkpoints") / qc.get("checkpoint", "best_egnn_pruning.pt")
    scorer, info = load_interface_scorer(checkpoint, torch_device=torch.device("cpu"), seed=int(provenance["seed"]))
    if info.status != "checkpoint_loaded":
        raise RuntimeError("EGNN checkpoint could not be loaded")
    scorer.eval()
    provider = PyRosettaRotamerProvider()
    failures = [row for row in provenance["failures"] if "candidate in each chi1 well" in row["error"]]
    print(f"Triwell failures: {len(failures)}; inspecting first {min(args.limit, len(failures))}")
    for failure in failures[:args.limit]:
        pdb = str(failure["pdb_id"]).lower()
        row = train[pdb]
        rel = Path(row["path"])
        if rel.is_absolute() or not (dataset / rel).resolve().is_relative_to(dataset.resolve()):
            raise ValueError(f"Unsafe graph path for {pdb}: {rel}")
        graph = load_graph(dataset / rel)
        active = select_ablation_active(
            graph, cal.get("selection_mode", "egnn"), int(cal["active_sites"]), int(provenance["seed"]), scorer,
            antigen_guidance_weight=float(qc["antigen_guidance_weight"]),
            antigen_proximity_scale=float(qc["antigen_proximity_scale_angstrom"]),
            contact_ca_cutoff=float(qc["contact_ca_cutoff_angstrom"]),
        )
        requested = {}
        for index in active.tolist():
            aa = AA[int(graph.x[index, :20].argmax())]
            key = (_THREE_LETTER[aa],
                   _nearest_dunbrack_bin(float(graph.backbone_phi[index])),
                   _nearest_dunbrack_bin(float(graph.backbone_psi[index])))
            requested[index] = key
        bins = provider.load_bins(set(requested.values()))
        print(f"\n{pdb}")
        for index, key in requested.items():
            raw = bins[key]
            raw_wells = {chi1_well_index(r.chi1_degrees) for r in raw if r.prior_probability > 0}
            retained = _dunbrack_templates_for_site(
                bins, AA[int(graph.x[index, :20].argmax())],
                float(graph.backbone_phi[index]), float(graph.backbone_psi[index]),
                probability_floor=float(rot["probability_floor"]),
                sigma_offsets=rot["sigma_offsets"],
            )
            retained_wells = {chi1_well_index(r.chi1_degrees) for r in retained}
            if len(retained_wells) == 3:
                continue
            best = {well: max((r.prior_probability for r in raw
                               if chi1_well_index(r.chi1_degrees) == well), default=0.0)
                    for well in range(3)}
            print(f"  {graph.residue_ids[index]} {key[0]} phi/psi={key[1:]}; "
                  f"raw wells={sorted(raw_wells)}, retained wells={sorted(retained_wells)}, "
                  f"best raw probability by well={best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
