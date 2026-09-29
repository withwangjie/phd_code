import csv
import subprocess
import sys
import time

from nanoqc.common.cpu_runtime import CPUStageMonitor
from nanoqc.common.stage_utilization import main, summarize


def test_cpu_monitor_attributes_the_stage_process_tree(tmp_path):
    path = tmp_path / "logs" / "busy.cpu.csv"
    # The first per-process reading primes its CPU counter. Windows process
    # discovery/load-average initialization can delay the next sample, so keep
    # the child alive long enough for a second measurement of the same process.
    burn = "import time\nend=time.time()+5.5\nwhile time.time()<end: pass\n"
    with CPUStageMonitor(path, enabled=True, interval=1) as monitor:
        child = subprocess.Popen([sys.executable, "-c", burn])
        monitor.watch(child.pid)
        child.wait()
        time.sleep(1.2)
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["status"] == "sample"]
    assert rows, "no samples recorded"
    logical = int(rows[0]["logical_cpus"])
    # One busy process is 100/logical percent of the host; allow scheduling slack.
    assert max(float(r["stage_cpu_percent"]) for r in rows) >= 0.5 * 100 / logical
    assert max(int(r["stage_processes"]) for r in rows) >= 1


def test_disabled_monitor_writes_nothing(tmp_path):
    with CPUStageMonitor(tmp_path / "x.cpu.csv", enabled=False) as monitor:
        monitor.watch(1)
    assert not (tmp_path / "x.cpu.csv").exists()


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_summary_flags_stages_below_target(tmp_path, capsys):
    cpu_fields = list(CPUStageMonitor.fields)
    blank = dict.fromkeys(cpu_fields, "")
    _write(tmp_path / "logs" / "qc_benchmark.cpu.csv", cpu_fields, [
        {**blank, "utc": "2026-09-29T00:00:00+00:00", "status": "sample", "host_cpu_percent": "85",
         "stage_cpu_percent": "82", "stage_rss_gib": "40"},
        {**blank, "utc": "2026-09-29T00:00:10+00:00", "status": "sample", "host_cpu_percent": "83",
         "stage_cpu_percent": "80", "stage_rss_gib": "41"}])
    _write(tmp_path / "logs" / "final_report.cpu.csv", cpu_fields, [
        {**blank, "utc": "2026-09-29T01:00:00+00:00", "status": "sample", "host_cpu_percent": "3",
         "stage_cpu_percent": "1.3"},
        {**blank, "utc": "2026-09-29T01:00:05+00:00", "status": "error", "error": "x"}])
    gpu_fields = ["utc", "status", "gpu_index", "gpu_uuid", "utilization_percent",
                  "memory_used_mib", "memory_total_mib", "power_watts", "error"]
    _write(tmp_path / "logs" / "structure_experiment.gpu.csv", gpu_fields, [
        {"utc": "2026-09-29T02:00:00+00:00", "status": "sample", "gpu_index": i, "gpu_uuid": "u",
         "utilization_percent": u, "memory_used_mib": "9000", "memory_total_mib": "15360",
         "power_watts": "60", "error": ""} for i, u in (("0", "91"), ("1", "88"))])
    by_stage = {e["stage"]: e for e in summarize(tmp_path, 80)}
    assert by_stage["qc_benchmark"]["status"] == "ok"
    assert by_stage["qc_benchmark"]["stage_cpu_percent_mean"] == 81.0
    assert by_stage["qc_benchmark"]["sampled_seconds"] == 10.0
    assert by_stage["final_report"]["status"] == "below"
    assert by_stage["structure_experiment"]["gpus"]["1"]["utilization_percent_mean"] == 88.0
    assert by_stage["structure_experiment"]["status"] == "ok"
    assert main([str(tmp_path), "--json", str(tmp_path / "u.json")]) == 0
    assert "| qc_benchmark |" in capsys.readouterr().out
    assert (tmp_path / "u.json").is_file()
