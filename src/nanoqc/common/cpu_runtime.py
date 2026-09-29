"""Read-only, optional stage CPU telemetry; never controls scheduling.

Each sample records whole-host CPU use and the CPU use of the stage's own
process tree (the orchestrated subprocess and every descendant, including
spawned workers). Both are percentages of the host's logical CPUs, the unit
top/htop report, so the stage share can be compared with a utilization target
while other users' load stays visible in the host column.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import threading

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is in requirements.txt
    psutil = None


class CPUStageMonitor:
    fields=("utc","status","host_cpu_percent","stage_cpu_percent","stage_processes",
            "logical_cpus","load_1min","ram_used_gib","ram_available_gib","stage_rss_gib","error")

    def __init__(self,path: Path,*,enabled: bool=False,interval: float=10):
        self.path=Path(path);self.enabled=enabled and psutil is not None
        self.unavailable=enabled and psutil is None
        self.interval=max(1,float(interval))
        self.stop=threading.Event();self.thread=None
        self.root_pid=None;self._processes={}

    def watch(self,pid: int) -> None:
        """Attribute the process tree rooted at ``pid`` to this stage."""
        self.root_pid=int(pid)

    def _stage_tree(self) -> tuple[float,int,float]:
        if self.root_pid is None:return 0.0,0,0.0
        try:
            root=psutil.Process(self.root_pid)
            tree=[root,*root.children(recursive=True)]
        except psutil.Error:
            return 0.0,0,0.0
        percent=0.0;rss=0;alive={}
        for process in tree:
            # Keep one Process object per PID: cpu_percent() measures since its previous call.
            tracked=self._processes.get(process.pid,process)
            try:
                percent+=tracked.cpu_percent(None);rss+=tracked.memory_info().rss
                alive[process.pid]=tracked
            except psutil.Error:
                continue
        self._processes=alive
        return percent,len(alive),rss/1024**3

    def sample(self) -> tuple:
        now=datetime.now(timezone.utc).isoformat()
        try:
            logical=psutil.cpu_count(logical=True) or 1
            host=psutil.cpu_percent(None)
            stage,count,rss=self._stage_tree()
            memory=psutil.virtual_memory()
            try:load=psutil.getloadavg()[0]
            except (AttributeError,OSError):load=""
            return (now,"sample",f"{host:.1f}",f"{stage/logical:.1f}",count,logical,
                    load if load=="" else f"{load:.2f}",f"{memory.used/1024**3:.2f}",
                    f"{memory.available/1024**3:.2f}",f"{rss:.2f}","")
        except (psutil.Error,OSError,ValueError) as exc:
            return (now,"error",*([""]*8),str(exc))

    def _record(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            if handle.tell()==0:writer.writerow(self.fields)
            psutil.cpu_percent(None)  # prime the host counter; the first reading is otherwise 0
            while not self.stop.wait(self.interval):
                writer.writerow(self.sample());handle.flush()
            writer.writerow(self.sample());handle.flush()

    def _run(self):
        try:self._record()
        except OSError:pass  # Monitoring failure cannot alter a scientific stage's outcome.

    def __enter__(self):
        if self.unavailable:
            try:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with self.path.open("a",newline="",encoding="utf-8") as handle:
                    writer=csv.writer(handle)
                    if handle.tell()==0:writer.writerow(self.fields)
                    writer.writerow((datetime.now(timezone.utc).isoformat(),"unavailable",
                                     *([""]*8),"psutil not installed"))
            except OSError:pass
        elif self.enabled:
            self.thread=threading.Thread(target=self._run,name="cpu-stage-monitor",daemon=True)
            self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        if self.thread:self.thread.join(timeout=4)
