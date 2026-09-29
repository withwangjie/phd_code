"""Read-only, optional stage GPU telemetry; never controls device scheduling."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import threading


class GPUStageMonitor:
    fields=("utc","status","gpu_index","gpu_uuid","utilization_percent",
            "memory_used_mib","memory_total_mib","power_watts","error")

    def __init__(self,path: Path,*,enabled: bool=False,interval: float=10):
        self.path=Path(path);self.enabled=enabled;self.interval=max(1,float(interval))
        self.stop=threading.Event();self.thread=None
        self.executable=shutil.which("nvidia-smi") if enabled else None

    def sample(self) -> list[tuple]:
        now=datetime.now(timezone.utc).isoformat()
        if not self.executable:return [(now,"unavailable",*([""]*6),"nvidia-smi not found")]
        try:
            result=subprocess.run([self.executable,
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3,check=False)
            if result.returncode:raise RuntimeError(result.stderr.strip() or f"exit={result.returncode}")
            rows=[tuple(item.strip() for item in row) for row in csv.reader(result.stdout.splitlines()) if row]
            if not rows or any(len(row)!=6 for row in rows):raise ValueError("Malformed GPU telemetry")
            return [(now,"sample",*row,"") for row in rows]
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
            return [(now,"error",*([""]*6),str(exc))]

    def _record(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            if handle.tell()==0:writer.writerow(self.fields)
            while True:
                writer.writerows(self.sample());handle.flush()
                if not self.executable or self.stop.wait(self.interval):return

    def _run(self):
        try:self._record()
        except OSError:pass  # Monitoring failure cannot alter a scientific stage's outcome.

    def __enter__(self):
        if self.enabled:
            self.thread=threading.Thread(target=self._run,name="gpu-stage-monitor",daemon=True)
            self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        if self.thread:self.thread.join(timeout=4)
