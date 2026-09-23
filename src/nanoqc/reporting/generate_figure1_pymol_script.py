"""Generate a traceable PyMOL Figure 1 script, without requiring PyMOL.

Default: choose a successful target nearest median size (prefer 10--13 sites),
independent of solver performance, and replay trained pruning/QUBO construction.
Explicit structure mode requires --active-residues and a CDR-H3 annotation.
Residue IDs use author chain:residue numbering, including insertion codes.
Only trusted project-generated .pt/checkpoint files may be supplied.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any, Sequence
from nanoqc.common.repo_io import sha256_file as sha256, REPO_ROOT, repo_path


def existing(paths: Sequence[Path], label: str) -> Path:
    """Return the first existing candidate, otherwise fail with useful context."""
    for path in paths:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'{label} not found: ' + ', '.join(map(str, paths)))


def residue_selection(ids: Sequence[str]) -> str:
    """Create object-scoped selectors, preserving chain and insertion codes."""
    terms = []
    for identifier in dict.fromkeys(ids):
        chain, resi = identifier.rsplit(':', 1)
        if not re.fullmatch(r'[A-Za-z0-9_]*', chain) or not re.fullmatch(r'-?\d+[A-Za-z]?', resi):
            raise ValueError(f'Unsupported/unsafe author residue identifier: {identifier!r}')
        terms.append(f'(chain "{chain}" and resi {resi})')
    return '(complex_obj and (' + ' or '.join(terms) + '))' if terms else 'none'


def choose_target(csv_path: Path, target: str | None) -> dict[str, str]:
    """Select deterministically by size, never by QAOA/SA outcome."""
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r.get('status') == 'success']
    if target:
        rows = [r for r in rows if r['target_id'] == target or r['pdb_id'].upper() == target.upper()]
        if len(rows) != 1:
            raise ValueError(f'Target matches {len(rows)} rows; supply a unique --target target_id.')
        return rows[0]
    if not rows:
        raise ValueError('No successful benchmark targets.')
    pool = [r for r in rows if 10 <= int(r['num_selected_sites']) <= 13] or rows
    med_bits = statistics.median(float(r['num_bits']) for r in rows)
    med_cdr = statistics.median(float(r['cdr3_len']) for r in rows)
    return min(pool, key=lambda r: (abs(float(r['num_bits']) - med_bits),
                                   abs(float(r['cdr3_len']) - med_cdr), r['target_id']))


def extract_source(source_id: str, data_root: Path, destination: Path) -> Path:
    """Resolve graph source, including ZIP members, without extracting archives wholesale."""
    parts = source_id.replace('\\', '/').split('::', 1)
    relative = Path(parts[0])
    direct = data_root / relative
    matches = [direct] if direct.is_file() else [p for p in data_root.rglob(relative.name)
               if p.as_posix().endswith('/' + relative.as_posix())]
    if len(matches) != 1:
        raise ValueError(f'Expected one raw source for {source_id}; found {len(matches)} under {data_root}')
    if len(parts) == 1:
        return matches[0].resolve()
    with zipfile.ZipFile(matches[0]) as archive:
        info = archive.getinfo(parts[1])
        if info.file_size > 200_000_000:
            raise ValueError('Structure archive member exceeds 200 MB limit.')
        destination = destination.with_suffix(Path(parts[1]).suffix)
        destination.write_bytes(archive.read(info))
    return destination.resolve()


def replay(args: argparse.Namespace, output: Path) -> tuple[Path, list[str], list[str], str, dict[str, Any]]:
    """Recover actual QUBO Active residues and CDR sequence from the trained pipeline."""
    import torch
    from nanoqc.data.safe_graph_load import load_graph
    from nanoqc.model.model_egnn_pruning import load_interface_scorer
    from nanoqc.model.model_egnn_pruning import extract_top_interface_subgraph
    from nanoqc.qubo.subgraph_to_qubo import InterfaceQUBOBuilder

    torch.set_num_threads(2)
    root = REPO_ROOT
    csv_path = existing([args.csv] if args.csv else [root / 'benchmark_results_500/snac_hard_qaoa_vs_sa_metrics.csv',
                        root / 'quantum-protein/benchmark_results_500/snac_hard_qaoa_vs_sa_metrics.csv'], 'Metrics CSV')
    row = choose_target(csv_path, args.target)
    project = csv_path.parent.parent
    graph_name = Path(row['source_file'].replace('\\', '/')).name
    graph_path = existing([args.graph] if args.graph else [Path(row['source_file']),
                         project / 'dataset_clean_500/graphs/test_snac_hard' / graph_name,
                         root / 'dataset_clean_500/graphs/test_snac_hard' / graph_name], 'Graph')
    checkpoint = existing([args.checkpoint] if args.checkpoint else [project / 'checkpoints_500/best_egnn_pruning.pt',
                          root / 'checkpoints_500/best_egnn_pruning.pt'], 'Trained checkpoint')
    manifest_path = csv_path.parent / 'run_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('run_manifest.json required to verify benchmark replay provenance.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('checkpoint_sha256') != sha256(checkpoint):
        raise ValueError('Checkpoint differs from the benchmark run.')
    for filename in ('model_egnn_pruning.py', 'subgraph_to_qubo.py', 'batch_benchmark_hard_set.py'):
        if manifest.get(filename) != sha256(repo_path(filename)):
            raise ValueError(f'{filename} differs from the benchmark; use matching project code.')
    graph = load_graph(graph_path)
    scorer, info = load_interface_scorer(checkpoint, torch_device=torch.device('cpu'), seed=42)
    if info.status != 'checkpoint_loaded':
        raise ValueError(f'Cannot render benchmark Active set with untrained weights: {info}')
    sub = extract_top_interface_subgraph(graph, scorer, probability_threshold=0.5,
                                        min_active=5, max_active=15, environment_radius=6.0)
    count = int(sub.is_active.sum())
    qubo = InterfaceQUBOBuilder(min_variables=min(20, 3 * count), max_variables=30, max_sites=15).build(sub)
    indices = sorted({int(record.original_node_index) for record in qubo.variable_map})
    active = [graph.residue_ids[i] for i in indices]
    if len(active) != int(row['num_selected_sites']) or len(qubo.physical_self) != int(row['num_bits']):
        raise ValueError('Replayed Active/bit counts differ from benchmark. Do not silently render a different instance.')
    vhh = [chain for chain, group in zip(graph.chain_ids, graph.chain_groups) if group == 0]
    if len(vhh) != 1:
        raise ValueError('Expected one annotated VHH chain.')
    source = args.structure or extract_source(graph.source_id, args.data_root or root / 'data', output / 'figure1_raw_source')
    metadata: dict[str, Any] = {'mode': 'benchmark_replay', 'target': row, 'metrics_csv': str(csv_path),
        'graph': str(graph_path), 'graph_sha256': sha256(graph_path), 'checkpoint_sha256': sha256(checkpoint),
        'source_id': graph.source_id, 'selection_rule': 'nearest median bits then CDR length; prefer 10-13 sites; outcome-independent',
        'num_bits': len(qubo.physical_self), 'variables': [vars(r) for r in qubo.variable_map],
        'expected_ca': {rid: graph.pos[i].tolist() for i, rid in enumerate(graph.residue_ids)},
        'expected_frozen': sorted(sub.residue_ids[i] for i in range(sub.num_nodes) if bool(sub.is_frozen_environment[i]))}
    return Path(source), active, vhh, graph.cdr3_seq, metadata


def prepare_structure(source: Path, destination: Path) -> tuple[dict[str, Any], dict[str, list[tuple[str, str]]]]:
    """Write a single-model mmCIF retaining author IDs and highest-occupancy atoms."""
    import gemmi
    structure = gemmi.read_structure(str(source))
    if not len(structure):
        raise ValueError('Structure has no models.')
    while len(structure) > 1:
        del structure[1]
    coordinates: dict[str, Any] = {}
    chains: dict[str, list[tuple[str, str]]] = {}
    for chain in structure[0]:
        for residue in chain:
            best: dict[str, Any] = {}
            for atom in residue:
                if atom.occ > 0 and (atom.name not in best or atom.occ > best[atom.name].occ):
                    best[atom.name] = atom.clone()
            for i in reversed(range(len(residue))):
                del residue[i]
            for atom in best.values():
                atom.altloc = '\x00'
                residue.add_atom(atom)
            tab = gemmi.find_tabulated_residue(residue.name)
            if 'CA' not in best or not tab.is_amino_acid():
                continue
            rid = f'{chain.name}:{str(residue.seqid)}'
            if rid in coordinates:
                raise ValueError(f'Ambiguous duplicate residue {rid}')
            pos = best['CA'].pos
            coordinates[rid] = (pos.x, pos.y, pos.z)
            chains.setdefault(chain.name, []).append((rid, tab.one_letter_code.upper()))
    if not coordinates:
        raise ValueError('No protein CA coordinates found.')
    structure.make_mmcif_document().write_file(str(destination))
    return coordinates, chains


def generate(args: argparse.Namespace) -> Path:
    """Generate PML, normalized structure and JSON provenance/selection manifest."""
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.active_residues:
        if not args.structure or args.vhh_chain is None:
            raise ValueError('Manual mode requires --structure and --vhh-chain.')
        source = args.structure.resolve()
        active = args.active_residues.split(',')
        vhh = [args.vhh_chain]
        cdr_seq = args.cdr3_seq or ''
        meta: dict[str, Any] = {'mode': 'explicit_residues', 'note': 'User-specified Active residues; QUBO membership not inferred.'}
    else:
        source, active, vhh, cdr_seq, meta = replay(args, output)
    normalized = output / 'figure1_structure.cif'
    ca, chains = prepare_structure(source, normalized)
    residue_selection(active)
    if any(r not in ca for r in active):
        raise ValueError('Active residues absent from structure: ' + str(set(active) - ca.keys()))
    if any(r.rsplit(':', 1)[0] not in vhh for r in active):
        raise ValueError('Active residues must belong to the annotated VHH chain.')
    for rid, xyz in meta.pop('expected_ca', {}).items():
        if rid not in ca or sum((a-b)**2 for a, b in zip(xyz, ca[rid])) > 0.01**2:
            raise ValueError(f'Graph/structure CA mismatch at {rid}; supply the original curated assembly.')
    if args.cdr3_residues:
        cdr = args.cdr3_residues.split(',')
    else:
        cdr_seq = args.cdr3_seq or cdr_seq
        if not cdr_seq:
            raise ValueError('Specify --cdr3-seq or --cdr3-residues; no numbering convention is assumed.')
        entries = chains.get(vhh[0], [])
        sequence = ''.join(aa for _, aa in entries)
        starts = [i for i in range(len(sequence)) if sequence.startswith(cdr_seq.upper(), i)]
        if len(starts) != 1:
            raise ValueError('CDR-H3 does not map uniquely to resolved VHH sequence; specify verified --cdr3-residues.')
        cdr = [rid for rid, _ in entries[starts[0]:starts[0]+len(cdr_seq)]]
    if any(r not in ca or r.rsplit(':', 1)[0] not in vhh for r in cdr):
        raise ValueError('CDR-H3 residues must exist on the VHH chain.')
    frozen = sorted(r for r, xyz in ca.items() if r not in active and any(
        sum((a-b)**2 for a, b in zip(xyz, ca[ar])) <= 36.0 for ar in active))
    expected = meta.pop('expected_frozen', None)
    if expected is not None and frozen != expected:
        raise ValueError('Frozen selection differs from graph replay; inspect structure/coordinate precision.')
    meta.update(source_structure=str(source.resolve()), source_sha256=sha256(source), active_residues=active,
                frozen_residues=frozen, cdr3_residues=cdr, active_cdr3_overlap=sorted(set(active) & set(cdr)),
                vhh_chains=vhh, normalized_structure_sha256=sha256(normalized),
                notes=['Rendered heavy atoms are from the input structure, not optimized rotamer output.',
                       'CDR-H3 and Active are distinct selections; tube width does not encode measured flexibility.',
                       'mode=2 contacts are geometric polar-contact candidates, not force-field terms or proven hydrogen bonds.',
                       'Frozen criterion is CA--CA <=6 Angstrom, not any-atom distance.',
                       'Active identities are replayed; CSV does not store the historical residue list.'])
    # Python block safely handles file paths with spaces; style uses ordinary PML commands.
    pml = f'''# Figure 1: input-structure illustration of Active/Frozen partition.
# Active is not synonymous with CDR-H3. Contacts are geometric candidates only.
python
from pathlib import Path
from pymol import cmd
figure_dir = Path({str(output)!r})
cmd.delete("complex_obj")
cmd.delete("contacts")
cmd.set("cif_use_auth", 1)
cmd.load(str(figure_dir / "figure1_structure.cif"), "complex_obj")
python end
hide everything, complex_obj
remove complex_obj and hydro
select vhh_chain, {residue_selection([rid for chain in vhh for rid, _ in chains[chain]])}
select antigen, complex_obj and polymer.protein and not vhh_chain
select active_res, {residue_selection(active)}
select cdr3_loop, {residue_selection(cdr)}
# byres expands the CA-based <=6 A neighborhood to complete residues.
select frozen_env, (byres ((complex_obj and polymer.protein and name CA) within 6.0 of (active_res and name CA))) and not active_res
select global_background, complex_obj and polymer.protein and not (active_res or frozen_env or cdr3_loop)
show cartoon, complex_obj and polymer.protein
color gray80, complex_obj
set cartoon_transparency, 0.45, complex_obj
set_color lightteal, [0.55, 0.72, 0.74]
set_color deepsalmon, [0.91, 0.46, 0.37]
set_color warmorange, [1.0, 0.46, 0.05]
color lightteal, frozen_env
show sticks, frozen_env and not hydro
set_bond stick_transparency, 0.60, frozen_env, frozen_env
set cartoon_color, lightteal, frozen_env
cartoon tube, cdr3_loop
set cartoon_tube_radius, 0.38
set cartoon_color, deepsalmon, cdr3_loop
set cartoon_transparency, 0.0, cdr3_loop
color deepsalmon, cdr3_loop and not frozen_env
show sticks, active_res and not hydro
set_bond stick_transparency, 0.0, active_res, active_res
set stick_radius, 0.20
color warmorange, active_res and elem C
color blue, (active_res or frozen_env) and elem N
color red, (active_res or frozen_env) and elem O
color yellow, (active_res or frozen_env) and elem S
select active_sidechains, active_res and not name N+CA+C+O+OXT and not hydro
distance contacts, active_sidechains, frozen_env and not hydro, cutoff=3.5, mode=2
set dash_radius, 0.05
set dash_color, yellow
set dash_gap, 0.25
hide labels
bg_color white
set orthoscopic, on
set ray_trace_mode, 1
set ray_shadows, 0
set antialias, 2
set ray_opaque_background, off
set opaque_background, off
viewport 1500, 1200
orient active_res
center active_res
zoom active_res, 6.0
deselect
python
assert cmd.count_atoms("active_res and name CA") == {len(active)}, "Active residue selection mismatch"
assert cmd.count_atoms("cdr3_loop and name CA") == {len(cdr)}, "CDR-H3 selection mismatch"
assert cmd.count_atoms("frozen_env and name CA") == {len(frozen)}, "Frozen selection mismatch"
cmd.save(str(figure_dir / "figure1_interface_pocket.pse"))
python end
ray 3000, 2400
python
cmd.png(str(figure_dir / "figure1_interface_pocket.png"), dpi=300, ray=0)
python end
'''
    pml_path = output / 'render_figure1.pml'
    pml_path.write_text(pml, encoding='utf-8')
    (output / 'figure1_selection_manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Generated {pml_path}\nActive={len(active)}, Frozen={len(frozen)}, CDR-H3={len(cdr)}, overlap={len(set(active)&set(cdr))}')
    return pml_path


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point. No QAOA optimization or PyMOL rendering is run here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--target', help='Unique target_id or unambiguous PDB ID')
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--structure', type=Path, help='PDB/CIF/CIF.GZ; benchmark mode verifies all graph CA coordinates')
    parser.add_argument('--vhh-chain')
    parser.add_argument('--active-residues', help='Explicit comma-separated chain:resi IDs, e.g. H:100,H:100A,H:101')
    parser.add_argument('--cdr3-seq')
    parser.add_argument('--cdr3-residues', help='Verified comma-separated chain:resi IDs; no guessed IMGT numbering')
    parser.add_argument('--out-dir', type=Path, default=Path('.'))
    args = parser.parse_args(argv)
    try:
        generate(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f'Figure generation failed: {error}\n')


if __name__ == '__main__':
    main()
