"""Wait for the current server training run, validate it, then benchmark."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    os.chdir(ROOT)
    pid = int(sys.argv[1])
    status = ROOT / 'logs/server_pipeline_status.json'
    def record(stage, **details):
        payload = dict(stage=stage, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **details)
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temp.replace(status)
        print(json.dumps(payload), flush=True)
    def run(stage, args, output):
        record(stage)
        with (ROOT / 'logs' / output).open('w') as log:
            subprocess.run([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        record('waiting_for_training', pid=pid)
        while Path(f'/proc/{pid}/stat').exists():
            if Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].startswith('Z'):
                break
            time.sleep(20)
        summary = json.loads(Path('checkpoints/training_summary.json').read_text())
        assert summary['status'] == 'complete', 'Training is not complete'
        assert summary['checkpoint']['strict_reload_verified']
        import hashlib
        checkpoint = Path('checkpoints/best_egnn_pruning.pt')
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary['checkpoint']['sha256']
        common = ['batch_benchmark_hard_set.py', '--backend', 'lightning.qubit',
                  '--torch-device', 'cpu', '--omp-threads', '3', '--shots', '1000',
                  '--qaoa-max-evals', '90', '--sa-reads', '50', '--sa-sweeps', '100']
        smoke = 'benchmark_results_server_smoke'
        run('benchmark_smoke', common + ['--workers', '2', '--max-targets', '3', '--out-dir', smoke], 'benchmark_smoke.log')
        with Path(smoke, 'snac_hard_qaoa_vs_sa_metrics.csv').open(encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3 and all(r['status'] == 'success' for r in rows), 'Smoke targets failed'
        run('benchmark_full', common + ['--workers', '4', '--out-dir', 'benchmark_results'], 'benchmark.log')
        with Path('benchmark_results/snac_hard_qaoa_vs_sa_metrics.csv').open(encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        successes = sum(r['status'] == 'success' for r in rows)
        record('complete' if successes == 400 else 'completed_with_failures', successful=successes, total=len(rows))
    except Exception as exc:
        record('failed', error=f'{type(exc).__name__}: {exc}')
        raise

if __name__ == '__main__':
    main()
