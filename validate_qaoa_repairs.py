"""Development-only, frozen-candidate ablations; never a held-out prediction test.

(P4 remediation) Two protocol-synchronization fixes relative to the earlier
revision of this script:

1. ``--output-dir`` is now an explicit CLI argument. Its default is no
   longer the fixed ``qaoa_repair_development`` -- it is a fresh,
   UTC-timestamped directory (:func:`_default_output_dir`), so two
   successive runs never share a directory unless the caller explicitly
   asks them to (by passing the same ``--output-dir`` both times).
2. Every per-``(seed, config)`` output pair (the reconstructed/relaxed
   ``.cif`` plus its ``.json`` detail report -- a "historical exact-mode
   benchmark" once written) is guarded: if either file already exists in
   ``--output-dir``, this run refuses to overwrite it and raises
   ``FileExistsError`` immediately, unless ``--overwrite`` is passed. This
   is on top of (not a replacement for) the pre-existing whole-run
   ``protocol.json`` marker, which still refuses to reuse a directory
   whose recorded protocol (scope/seeds/budgets/configs/code hashes)
   differs from this run's.

Every per-``(seed, config)`` JSON report and the run-level manifest now
also explicitly carry the full measurement ledger read straight off the
``OptimizationResult`` actually returned by ``optimize_robust`` --
``eval_shots``, ``cvar_alpha``, ``total_opt_shots``, ``output_shots``,
``optimizer_success`` and ``termination_reason`` -- rather than only the
nominal *input* protocol values, so a reader can see exactly what
measurement budget and convergence outcome each individual run achieved
(these can legitimately differ run-to-run even at fixed nominal settings,
since ``nfev``/``total_opt_shots`` depend on how many evaluations each
restart's own COBYLA call actually consumed before stopping).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from batch_benchmark_hard_set import _time_budget_counts
from qaoa_interface_sampler import XYMixerQAOASampler
from subgraph_to_qubo import AllAtomInterfaceQUBOBuilder, evaluate_atomistic_prediction

# (Version Discrepancy remediation) Every root-directory module this script's
# reproducibility protocol depends on -- including evaluate_complex_metrics.py,
# now that evaluate_atomistic_prediction delegates its Fnat/iRMSD/LRMSD/DockQ/
# clash computation to it (see subgraph_to_qubo.py) -- is hashed into the
# protocol manifest below, so a silent change to any of them is detected by
# the existing protocol.json marker check exactly like a change to this
# script itself.
_TRACKED_MODULES: tuple[str, ...] = (
    "qaoa_interface_sampler.py", "validate_qaoa_repairs.py", "subgraph_to_qubo.py",
    "batch_benchmark_hard_set.py", "evaluate_complex_metrics.py",
)


def _default_output_dir() -> Path:
    """A fresh, collision-free default output directory, timestamped to the second (UTC).

    Replaces the old hardcoded ``qaoa_repair_development`` default, which
    let successive runs silently share (and, via the per-file overwrite
    guard's absence, previously could clobber) one directory's per-``(seed,
    config)`` CIF/JSON outputs. Reusing a specific directory -- including
    the historical ``qaoa_repair_development`` name -- is still possible by
    passing ``--output-dir`` explicitly; it remains protected by both the
    ``protocol.json`` marker and the per-file overwrite guard in
    :func:`main`.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"qaoa_repair_development_{stamp}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for this development run. Defaults to a fresh, "
             "UTC-timestamped directory (see _default_output_dir) so successive "
             "runs never share, and cannot silently overwrite, one another's "
             "per-(seed, config) output files. Pass an existing directory "
             "explicitly to resume/extend it -- still guarded by the "
             "protocol.json marker and --overwrite below.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting an existing (seed, config) CIF/JSON output pair "
             "in --output-dir. Without this flag, an existing output file for a "
             "(seed, config) this run is about to produce raises FileExistsError "
             "immediately instead of silently clobbering a historical benchmark "
             "result.",
    )
    return parser


def _reject_existing_outputs(destination: Path, json_path: Path, *, overwrite: bool) -> None:
    """Refuse to silently clobber a historical (seed, config) benchmark result.

    Args:
        destination: The relaxed-structure ``.cif`` path this run is about
            to write (plus its ``relax_positions``-generated
            ``<stem>_discrete.cif`` sibling, also checked).
        json_path: The per-``(seed, config)`` JSON detail-report path this
            run is about to write.
        overwrite: When ``True``, skip the check entirely (explicit opt-in).

    Raises:
        FileExistsError: If ``overwrite`` is ``False`` and any of
            ``destination``, its ``_discrete`` sibling, or ``json_path``
            already exists.
    """
    if overwrite:
        return
    discrete_sibling = destination.with_name(destination.stem + "_discrete.cif")
    existing = [p for p in (destination, discrete_sibling, json_path) if p.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing historical benchmark output {existing}; "
            "pass --overwrite to allow this explicitly, or use a different --output-dir"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argument_parser().parse_args(argv)
    root = Path('real_complex_pilot_v3')
    out = args.output_dir if args.output_dir is not None else _default_output_dir()
    out.mkdir(exist_ok=True, parents=True)
    eval_shots = 500
    cvar_alpha = .1
    output_shots = 1000
    protocol = dict(
        scope='4S10 development, not independent validation', seeds=[42, 43, 44],
        max_evals=360, output_shots=output_shots, eval_shots=eval_shots, cvar_alpha=cvar_alpha,
        configs=[
            ['mean_single', 'mean', 1, 'max_coefficient'],
            ['mean_multi', 'mean', 4, 'max_coefficient'],
            ['cvar_multi', 'cvar', 4, 'max_coefficient'],
            ['cvar_iqr', 'cvar', 4, 'feasible_iqr']],
        code_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in _TRACKED_MODULES},
    )
    marker = out / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Protocol changed: preserve this run and use a new output directory')
    marker.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows = []
    for seed in protocol['seeds']:
        folder = root / 'results/4s10' / f'seed_{seed}' / 'experiment'
        saved = json.loads((folder / 'completed.json').read_text())
        for name, digest in saved['artifacts'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        case = json.loads((folder.parent / 'experiment.json').read_text())
        mapping = json.loads((folder / 'allatom_mapping.json').read_text())
        qubo = np.load(folder / 'allatom_qubo.npz'); coordinates = np.load(folder / 'candidate_coordinates.npz')
        rmsd = {r['bits']: float(r['rmsd']) for r in csv.DictReader((root / 'diagnostics_4s10' / f'states_{seed}.csv').open())}
        sampler = XYMixerQAOASampler(qubo['physical_self'], qubo['physical_pair'],
            {int(k): v for k, v in mapping['site_to_variables'].items()}, simulation_mode='subspace', seed=seed)
        preprocess = time.perf_counter(); ground = sampler.enumerate_ground_states(); energy = sampler.feasible_energy_map()
        preprocessing_seconds = time.perf_counter() - preprocess
        # Reuse frozen atom coordinates; reject any topology/preparation mismatch.
        builder = AllAtomInterfaceQUBOBuilder(Path(case['input_structure']), case['active_residues'], seed=seed)
        if not np.allclose(builder.base_positions, coordinates['base_positions_nm'], atol=1e-6, rtol=0):
            raise ValueError('Prepared coordinates differ from frozen experiment')
        for name, objective, restarts, scale in protocol['configs']:
            prefix = out / f'seed_{seed}_{name}'
            destination = prefix.with_suffix('.cif')
            json_path = prefix.with_suffix('.json')
            _reject_existing_outputs(destination, json_path, overwrite=args.overwrite)
            start = time.perf_counter()
            opt = sampler.optimize_robust(
                max_evals=protocol['max_evals'], restarts=restarts, objective=objective,
                cvar_alpha=protocol['cvar_alpha'], parameter_scale=scale, eval_shots=protocol['eval_shots'],
            )
            sampled = sampler.sample(opt, shots=protocol['output_shots'], ground_state=ground)
            elapsed = time.perf_counter() - start
            probabilities = np.abs(sampler.subspace_state(np.r_[opt.gammas, opt.betas])) ** 2
            bits = sampler._subspace_bits
            near = np.array([rmsd[''.join(map(str, b))] <= 1. for b in bits])
            gm = np.array([abs(energy[tuple(b)] - ground.energy) <= 1e-6 for b in bits])
            counts = dict(sampled.counts)
            selected = min(counts, key=lambda b: (energy[b], b))
            positions = coordinates['base_positions_nm'].copy()
            for variable in np.flatnonzero(selected):
                positions[coordinates[f'indices_{variable}']] = coordinates[f'positions_nm_{variable}']
            expected = energy[selected] + mapping['metadata']['physical_constant_offset']
            assert np.isclose(builder.energy(positions), expected, atol=1e-4, rtol=1e-9)
            relaxation = builder.relax_positions(positions, destination, minimize_iterations=200)
            accuracy = evaluate_atomistic_prediction(Path(case['reference_structure']), destination,
                active_residues=case['active_residues'], alignment_residues=case['alignment_residues'],
                partner_residues=case['partner_residues'])
            # (P4 remediation) Full measurement ledger, read off the actual
            # OptimizationResult rather than only the nominal input protocol --
            # nfev/total_opt_shots and the honest convergence classification can
            # legitimately vary run-to-run even at fixed nominal settings.
            measurement_ledger: Dict[str, Any] = dict(
                eval_shots=opt.eval_shots, cvar_alpha=protocol['cvar_alpha'],
                total_opt_shots=opt.total_opt_shots, output_shots=protocol['output_shots'],
                nfev=opt.nfev, optimizer_success=opt.optimizer_success,
                termination_reason=opt.termination_reason,
            )
            row = dict(seed=seed, config=name, evaluations=opt.evaluations, objective=opt.objective_name,
                objective_value=opt.objective_value, mean_energy=opt.energy,
                gap=energy[selected] - ground.energy, p_ground=float(probabilities[gm].sum()),
                p_rmsd_le_1=float(probabilities[near].sum()),
                rmsd_before=rmsd[''.join(map(str, selected))], rmsd_after=accuracy['sidechain_rmsd_angstrom'],
                solver_seconds=elapsed, shared_enumeration_seconds=preprocessing_seconds,
                eval_shots=measurement_ledger['eval_shots'], cvar_alpha=measurement_ledger['cvar_alpha'],
                total_opt_shots=measurement_ledger['total_opt_shots'], output_shots=measurement_ledger['output_shots'],
                optimizer_success=measurement_ledger['optimizer_success'],
                termination_reason=measurement_ledger['termination_reason'])
            baselines = []
            for method in ('sa', 'uniform', 'greedy'):
                c, seconds, queries = _time_budget_counts(sampler, method, elapsed, seed + 101, 100, 50)
                best = min(c, key=lambda b: (energy[b], b))
                baselines.append(dict(method=method, seconds=seconds, energy_queries=queries, outputs=sum(c.values()),
                    gap=energy[best] - ground.energy, rmsd_before=rmsd[''.join(map(str, best))]))
            detail = dict(metrics=row, measurement_ledger=measurement_ledger, baselines=baselines,
                selected_bits=list(selected), counts=[dict(bits=list(b), count=c) for b, c in counts.items()],
                optimization=dict(gammas=opt.gammas.tolist(), betas=opt.betas.tolist(), history=opt.history,
                    restarts=opt.restart_records, optimizer_success=opt.optimizer_success,
                    termination_reason=opt.termination_reason, shot_ledger=opt.shot_ledger),
                relaxation=relaxation, structure=accuracy,
                source_manifest_sha256=hashlib.sha256((folder / 'completed.json').read_bytes()).hexdigest())
            json_path.write_text(json.dumps(detail, indent=2), encoding='utf-8')
            rows.append(row)
            with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows); handle.flush()
            print(seed, name, 'gap', row['gap'], 'p_ground', row['p_ground'], 'RMSD', row['rmsd_after'],
                  'optimizer_success', row['optimizer_success'], 'termination_reason', row['termination_reason'],
                  flush=True)
    lines = ['# QAOA 修复开发验证', '',
        f'输出目录：{out}。4S10 已用于开发，不能作为独立验证。全部配置每次总预算 {protocol["max_evals"]} 次评估、'
        f'{protocol["output_shots"]} 次输出（eval_shots={protocol["eval_shots"]}, cvar_alpha={protocol["cvar_alpha"]}）；'
        '原实验只有 90 次，直接比较不能视为等预算收益。',
        'CVaR 使用精确合法子空间概率。经典对照共享已枚举能量表，按每个量子配置的优化加采样 CPU 时间执行软截止预算；预处理单独记录。不是量子硬件加速实验。',
        '输出结构按原始物理能量选取，参考 RMSD 仅用于事后评估。', '',
        '|种子|配置|评估数|基态概率|所选能隙|最终侧链 RMSD Å|optimizer_success|termination_reason|',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(
            f"|{r['seed']}|{r['config']}|{r['evaluations']}|{r['p_ground']:.4%}|{r['gap']:.4f}|{r['rmsd_after']:.4f}|"
            f"{r['optimizer_success']}|{r['termination_reason']}|"
        )
    lines += ['', '同时间经典对照明细保存在逐配置 JSON 中。全部参数选择属于开发探索；需要冻结配置后在新的独立队列上验证。',
        '每个 (种子, 配置) 的完整测量台账（eval_shots/cvar_alpha/total_opt_shots/output_shots/termination_reason）'
        '见逐配置 JSON 的 measurement_ledger 字段与本目录的 metrics.csv。']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out / 'REPORT.md')


if __name__ == '__main__':
    main()
