"""Multi-GPU dispatch must preserve the frozen target and calibration order."""

import argparse
import csv
import json
import os
from pathlib import Path

from nanoqc.experiments import generate_energy_calibration_dataset as calibration
from nanoqc.experiments import run_real_complex_pilot as pilot


def test_calibration_shards_merge_in_training_manifest_order(tmp_path, monkeypatch):
    def fake_shard(argv, device, log_path):
        shard=int(argv[argv.index("--shard-index")+1])
        csv_path=Path(argv[argv.index("--out-csv")+1])
        provenance_path=Path(argv[argv.index("--out-provenance")+1])
        with csv_path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=["training_row_index","assignment_index","device"])
            writer.writeheader()
            writer.writerow({"training_row_index":shard,"assignment_index":0,"device":device})
        provenance_path.write_text(json.dumps({
            "complexes_discovered":1,"complexes_attempted":1,"complexes_succeeded":1,
            "rows_written":1,"failures":[],"input_quality_exclusions":[],
        }),encoding="utf-8")
        return 0

    monkeypatch.setattr(calibration,"_calibration_shard",fake_shard)
    args=argparse.Namespace(out_csv=tmp_path/"train.csv",out_provenance=None,
        workers=2,gpu_devices=["0","1"])
    assert calibration._run_parallel_shards(args)==0
    with args.out_csv.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    assert [(row["training_row_index"],row["device"]) for row in rows]==[("0","0"),("1","1")]
    provenance=json.loads((tmp_path/"train.provenance.json").read_text(encoding="utf-8"))
    assert provenance["complexes_succeeded"]==2
    assert provenance["parallel_workers"]==2
    assert provenance["csv_sha256"]==calibration.sha256(args.out_csv)


def test_structural_worker_sets_openmm_device_before_solver(monkeypatch):
    from nanoqc.experiments import batch_benchmark_hard_set as benchmark
    seen=[]
    monkeypatch.setattr(benchmark,"_recovery_benchmark_main",
        lambda argv: seen.append((list(argv),os.environ["QP_OPENMM_DEVICE"])) or 0)
    monkeypatch.setenv("QP_OPENMM_DEVICE","0")
    assert pilot._run_recovery_on_device(["--manifest","frozen.json"],"1")==0
    assert seen==[(["--manifest","frozen.json"],"1")]
