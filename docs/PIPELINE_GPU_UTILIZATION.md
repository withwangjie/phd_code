# GPU utilization across the formal pipeline

## Two T4 / 80-core / 256-GB execution profile

| Phase | Execution | GPU use |
|---|---|---|
| Data audit / sequence independence / graph construction | CPU pools, separate memory-heavy caps | CPU parsing and alignment |
| Queue-freeze candidate preparation | 8 spawned workers, 4 per GPU | OpenMM compatibility energies |
| EGNN training | 2 DDP ranks, FP32; global batch 4 | Both GPUs |
| Formal Amber diagnostic / energy calibration | 8 processes, 4 per GPU | OpenMM energies |
| Full training dual-energy diagnosis | 8 processes, 4 per GPU | OpenMM energies and relaxation |
| Structural candidate preparation | 8 spawned workers, 4 per GPU | OpenMM compatibility energies |
| Structural target evaluation | Up to 8 spawned workers, 4 per GPU | OpenMM energies and relaxation |
| QAOA feasible-subspace / SA / coarse sensitivity / external coarse validation | CPU simulation pools | CPU by design |
| Statistics / reporting / FASPR / Phenix | CPU tools and pools | CPU by design |

The profile is a starting configuration, not a measured optimum. Preparing a
candidate includes CPU structure parsing, Rosetta rotamers, CPU EGNN ranking
and deterministic Reference-platform hydrogen placement. Concurrent preparation
can overlap these tasks with independent GPU energy calculations. Keeping these
steps on their existing numerical backends avoids silently changing selected
sites or prepared coordinates. No change to batch size, AMP, force field,
precision, sampling, solver budgets or physical acceptance is made here.

## Ordered admission with concurrent preparation

Candidate preparation is independent of admission. Submit the first occurrence
of each candidate PDB to a fixed-device worker after skipping explicit/training
PDB and training-cluster overlaps. Consume preparations in the original
candidate order. Training and already-selected sequence/cluster exclusions remain
parent decisions in that order; completion speed never determines admission.
Preserve preparation failures and defer physical-check failures until their
original position after sequence screening. Prospective preparations of candidates
later excluded for selected-target overlap are not admitted or solver results.
Positive target caps can leave such preparation artifacts; only
`selected_targets.json` and `eligibility.json` define the selected denominator.

Preparation workers finish or cancel queued work before recovery workers start.
Individual worker processes are assigned fixed CUDA devices. Per-target seeds,
ordered result merge and failed-target accounting are unchanged. CUDA bitwise
equivalence and throughput remain server verification tasks.

## Runtime evidence

The profile enables ten-second read-only NVIDIA queries during each orchestrated
subprocess. `logs/<stage>.gpu.csv` records timestamps, both GPU UUIDs, utilization,
used/total memory and power. Query errors are recorded and never change a
scientific stage's status. These measurements cover all activity on each device,
not exclusively the pipeline; short phases may have few samples.

`prepared/<pdb>/preparation_execution.json` records the PID, assigned CUDA index,
wall time and preparation/physical failures. The parent
`preparation_execution.json` records submitted candidates and the device policy;
`structure_execution.json` records recovery worker assignment. Compare completed
complexes per hour, stage wall time, failure counts and memory, not utilization
percentage alone.

## CPU pool sizing and the 80 % target (A34)

Independent CPU pools (data audit, QC benchmark and the QC cases that
sensitivity, exploration and external validation split among themselves) are
sized to `target_cpu_utilization: 0.80` of the **logical** CPUs the process may
use, i.e. the unit top/htop report: 80 logical CPUs give 64 one-thread workers
whether they are 80 physical cores or 40 cores with two hardware threads. The
count honors CPU affinity and a cgroup v2 quota, keeps `cpu_reserve_cores` free
and stops at `worker_ram_gb` per worker above the free-RAM floor. Check the
resolved values in `.runtime/server_resolution.json`: `cpu_count_basis`,
`cpu_counted`, `cpu_physical_cores`, `qc_workers` and `cpu_pool_utilization`.

Expected CPU share by phase when the pools are full:

| Phase | Expected CPU share | Why |
|---|---|---|
| Data audit, QC benchmark, sensitivity/exploration QC cases | about 80 % | pool of 0.8 x logical CPUs; fewer tasks than workers lowers it |
| Graph construction | low | thread pool under the Python GIL |
| EGNN training | low CPU; GPU bound by the frozen global batch of 4 | batch size is a protocol setting |
| OpenMM preparation / calibration / structural targets | about 10 % CPU; GPU is the bottleneck | 8 processes, double precision on T4 |
| Sequential admission, statistics, reporting | low | short and ordered by design |

## Measuring utilization per stage

With `cpu_monitor_enabled` and `gpu_monitor_enabled`, every orchestrated
subprocess writes `logs/<stage>.cpu.csv` (host CPU %, the stage process tree's
CPU % of all logical CPUs, process count, load, RAM) next to
`logs/<stage>.gpu.csv`. Summarize a running or finished run with:

```bash
python -m nanoqc.common.stage_utilization <run_dir> --target 80
```

It prints each stage's sampled time, mean stage and host CPU %, peak stage
RSS and each GPU's mean utilization and peak memory, marking stages whose CPU
share and every GPU stay below the target. Raise GPU per-device workers only
after this table shows GPU headroom and peak memory well under 15 GiB, and
record the change as a protocol amendment.

## Optional MPS

Several processes on a device do not guarantee simultaneous CUDA kernels.
[NVIDIA's MPS architecture documentation](https://docs.nvidia.com/deploy/mps/architecture.html)
explains time slicing without MPS and concurrent client scheduling with MPS.
An administrator can evaluate MPS for small independent systems; this code does
not start or stop a service, change compute mode or remap visible devices.
Record an MPS change with the runtime configuration and measure throughput.

OpenMM double precision is retained. The
[OpenMM platform properties](https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html)
describe faster mixed/single alternatives, their numerical differences and
synchronization options. Switching precision requires a separate equivalence
study and protocol decision; CPU spin waiting is not used to manufacture load.

## Deployment

Pull the updated branch, then launch a new formally resolved run using the
existing deployment entry point. A previously frozen run does not acquire these
worker counts by pulling code: config and code hashes still protect resume.
Do not delete old checkpoints, audits or outcome records to bypass that contract.

```bash
cd /data/phd_code
git pull --ff-only origin claude/blissful-heisenberg-3n2jf3
./scripts/deploy_launch.sh
```

No server run or speedup measurement was performed during the local implementation.
