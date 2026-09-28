# Full training structure and dual-energy diagnosis

This is an assessment of **every training-manifest entry**, including entries
that cannot be prepared. It does not fit coefficients, change solver settings or
consume validation/test graphs. It reads a frozen run and writes a separate
directory. The earlier formal report, calibration CSV and checkpoint are kept.

## Server command

After synchronizing the implementation to the server:

```bash
cd /data/phd_code
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export QP_OPENMM_PLATFORM=CUDA
export QP_OPENMM_PRECISION=double

python -m nanoqc.experiments.training_energy_diagnostic \
  --run-dir /data/phd_code/runs/experiments_full_run_20260926_060041 \
  --out-dir /data/phd_code/runs/training_dual_energy_$(date +%Y%m%d_%H%M%S) \
  --workers 8 --workers-per-gpu 4 --gpu-devices 0 1 --iterations 200
```

Use the actual completed training run as `--run-dir`. Its dataset manifest,
checkpoint, family map, scientific configuration, rotamer version and seed stream
are the sources. Eight workers share the two CUDA devices, four per device,
with one BLAS/OpenMP thread each. The worker-to-device map is recorded in
provenance; random sampling uses the global training row, not the worker index.
Explicit CUDA
selection fails on CUDA problems; it does not silently fall back to CPU.

This is a starting configuration for the 80-core/256-GB/two-T4 host, not a
measured optimum. Shared workers can overlap CPU preparation with GPU work.
Without NVIDIA MPS, different CUDA contexts use GPU time slices; MPS can permit
concurrent kernels when a single client underuses the GPU. See NVIDIA's
[architecture](https://docs.nvidia.com/deploy/mps/architecture.html) and
[usage guidance](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).
The program does not start an MPS service. Compare completed complexes/hour,
GPU memory and RAM at 2, 4 and 8 workers before increasing concurrency further.
The server CPU task cap is now 64; graph and memory-heavy task caps remain
separate. Existing frozen formal runs keep their resolved runtime configuration.

The output must be an empty directory outside the source run. The command rejects
overwriting a previous diagnostic. This initial implementation has no resume
mode. Each assigned complex prints progress to its persistent
`worker_logs/gpu_*_shard_*.log`; the parent records its merge result or errors in
`generation.log`. Per-state artifacts are written as work proceeds.

## What “full” means

- Force `max_complexes=0`, even if the source run's calibration had a target cap.
- Visit every entry whose frozen split is `train`, in deterministic manifest order.
- Use the frozen `assignments_per_complex` value, normally **64**, for each
  constructible complex: half the lowest coarse-energy states and half unique
  uniformly sampled legal states, retaining the deterministic coarse minimum as
  anchor. This is full coverage of training complexes, **not exhaustive sampling
  of all 729 legal six-site/three-state assignments**.
- Keep the same full chi1..chiN assignment for coarse and atomistic measurements.
- Evaluate all planned sampled states, regardless of raw clashes or energy.
  If one state's relaxation fails, preserve its raw measurement and continue.
- Record preparation exclusions and complex construction failures. They remain
  in the training denominator; they cannot be replaced by another complex.

For the previously reported 235 training entries and 64 assignments each, the
nominal upper bound is 15,040 sampled states. Actual state count is lower when
structures/candidates cannot be prepared; the report gives those reasons.

## Measurements

For every evaluated state: save raw full coordinates, relaxed coordinates when
available, exact chis, graph and coordinate hashes, raw/relaxed Amber potential,
force-group energies, geometry audits and measured movable-atom residual force.
Source preparation is audited before attempting coarse construction so a later
QUBO failure does not erase an available preparation record.

Both energy differences use assignment zero as a fixed anchor. If the anchor's
energy is unavailable, the corresponding deltas remain unavailable; absolute
measurements for other states are retained. No better-looking anchor is substituted.

The audit uses A28's extreme-distance and residual-force definitions. Passing the
distance floor is not full packing or stereochemical validation. No row is
removed because it has an unusually high energy.

## Outputs

| Artifact | Content |
|---|---|
| `diagnostic_manifest.json` | Source-run hash, full source-module hashes, command and runtime platform |
| `dual_energy_train.csv` | Every sampled state's raw/relaxed energies, deltas, statuses and failures |
| `dual_energy_train.provenance.json` | Source/checkpoint/cluster/CSV hashes, seeds and training denominator |
| `dual_energy_summary.json` | Closed complex/state denominators, exclusions, failures and per-complex ranks |
| `dual_energy_report.md` | Readable descriptive report |
| `structures/train_*` | Preparation, coarse instance, per-state JSON/CIF artifacts and failure records |
| `worker_logs/gpu_*_shard_*.log` | Live per-GPU progress, retained after completion |

The summary shows coarse-versus-raw and coarse-versus-relaxed Spearman ranks for
all finite states and for the explicitly defined screened subsets separately.
Missing and constant correlations remain `n/a`. These are descriptive training
diagnostics; repeated states are not independent proteins and screened-subset
agreement does not establish surrogate validity over the full search space.

The calibration fitter rejects CSVs marked as dual-energy diagnostics. Reaching
the end of this command means assessment has finished and its denominators close;
it does not mean every structure converged or an Amber calibration passed.

## Next decision

Read preparation failure patterns, hydrogen-aware geometry, nonbonded energy
contributions, convergence distributions and within-complex ranks before deciding
whether to revise candidate preparation or attempt a separate family-grouped fit.
Retain the current failed diagnostic calibration and unused coefficients.
