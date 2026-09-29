r"""Fast, reproducible coordinate audit. Requires gemmi, numpy and scipy.

Run (from the repository root, with src/ on PYTHONPATH): python -m nanoqc.data.audit_all_datasets
No source structures are modified. Multi-model files use model 1 only.
"""
from __future__ import annotations
import argparse
import ast
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing as mp
import pathlib
import re
import time
import zipfile
import gemmi
import numpy as np
from scipy.spatial import cKDTree
from nanoqc.common.repo_io import REPO_ROOT, sha256_file, iter_jsonl
from nanoqc.common.device_errors import raise_if_resource_error

# Definitions now live in focused modules; re-exported so every existing
# `from nanoqc.data.audit_all_datasets import ...` keeps working.
from nanoqc.data.audit_structures import (  # noqa: E402,F401
    ZIP_LOCAL,
    BACKBONE,
    SIDECHAIN_HEAVY,
    pdb_compat,
    ASSEMBLY_SUBSETS,
    STRUCTURE_SOURCE_KEY,
    _PREBUILT_ASSEMBLY,
    prebuilt_assembly,
    biological_assembly_structure,
    materialize_graph_complex,
    read_structure,
    _read_raw_structure,
    chain_data,
)

BASE = REPO_ROOT  # standalone-run defaults (data/, outputs) are relative to the checkout root
ANNOTATIONS = {}
PDB_ANNOTATIONS = collections.defaultdict(list)
CHAIN_ANNOTATIONS = {}
# SAbDab metadata is kept independently from SNAC annotations so a formal
# sabdab_vhh entry never passes merely because the same PDB is also present in
# SNAC. This is the source-of-truth for SAbDab H/L/antigen chain identities.
SABDAB_ANNOTATIONS = collections.defaultdict(list)
# Entry-level resolution by PDB ID from curation metadata (SNAC curation
# summaries, SAbDab summary tables). SNAC's curated complex files carry no
# REMARK 2, so the structure file alone reports no resolution for any of them.
RESOLUTION_BY_PDB = {}
# Metadata files the resolution fallback read, with their hashes: the gate's
# outcome depends on them, so a run must be able to name them afterwards.
RESOLUTION_SOURCE_FILES = []
# Pipeline staging under the data root, never study input: the external-VHH
# preparation writes thousands of antigen-only structures (Foldseek input),
# downloaded candidates and its own copies of summary tables there. They would
# otherwise enter the audit as 'extra_external_vhh' rows and, worse, decide the
# resolution gate from a file no study step declares.
PIPELINE_WORK_DIRS = frozenset({'external_vhh'})
# Only curated VHH-antigen sources participate in the formal structural
# similarity universe. Generic RCSB complexes remain auditable but are not
# formal VHH training data.
FORMAL_SOURCE_PRIORITY = ('snac_db', 'sabdab_vhh', 'train_rcsb')
FORMAL_VHH_SUBSETS = frozenset({'snac_db', 'sabdab_vhh'})
INTERFACE_CONTACT_CUTOFF_ANGSTROM = 4.5
MAX_RESOLUTION_ANGSTROM = 3.0
# Conservative absolute floor for distinct-residue protein heavy atoms;
# this is a coordinate-overlap screen, not a MolProbity clashscore.
MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM = 1.0
MIN_INTERFACE_OCCUPANCY = 0.90
ALLOW_INTERFACE_ALTLOC = False
REQUIRE_RESOLUTION = True
REQUIRE_COMPLETE_INTERFACE_SIDECHAINS = True

def literal(s, default):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return default

def parse_resolution(value):
    """Worst (largest) positive number in a metadata field; None when it states none.

    A SAbDab row may list one value per deposited entry ('2.5, 2.7') [R33]. The
    gate is an upper bound, so the worst value is the conservative reading;
    taking the first would let a field order decide admission. Fields that
    state no resolution ('Resolution is Missing', 'NOT', 'NA') give None.
    """
    values = [float(m) for m in re.findall(r'\d+(?:\.\d+)?', str(value or ''))]
    values = [v for v in values if math.isfinite(v) and v > 0]
    return max(values) if values else None


def _record_resolution(pdb, value, source):
    resolution = parse_resolution(value)
    pdb = str(pdb or '').strip().upper()
    if resolution is not None and re.fullmatch(r'[0-9A-Z]{4}', pdb):
        RESOLUTION_BY_PDB.setdefault(pdb, (resolution, source))


def _staging(root, path):
    """True for a file under a pipeline working directory (never study input)."""
    return path.relative_to(root).parts[0] in PIPELINE_WORK_DIRS


def _record_source_file(root, path, kind):
    RESOLUTION_SOURCE_FILES.append(dict(path=str(path.relative_to(root)), kind=kind,
                                        sha256=sha256_file(path)))


def load_annotations(root):
    ANNOTATIONS.clear()
    PDB_ANNOTATIONS.clear()
    CHAIN_ANNOTATIONS.clear()
    SABDAB_ANNOTATIONS.clear()
    RESOLUTION_BY_PDB.clear()
    RESOLUTION_SOURCE_FILES.clear()
    for path in sorted(root.rglob('*_curation_summary.csv')):
        if path.parent.name not in ('curated_structures', 'benchmark_dataset') or _staging(root, path):
            continue
        _record_source_file(root, path, 'snac_curation_summary')
        for row in csv.DictReader(path.open(encoding='utf-8-sig', newline='')):
            ANNOTATIONS[(str(path.parent), row['Name'])] = row
            _record_resolution(row.get('PDB_ID'), row.get('Resolution'), 'snac_curation_summary')
            for field, old_id, typ in [('VH', 'Chain_VH_old_id', 'VHH' if path.name.startswith('nb_') else 'VH'), ('VL', 'Chain_VL_old_id', 'VL')]:
                seq = row.get('Sequence_' + field, '')
                if not seq or seq.lower() == 'nan':
                    continue
                if row.get('TCR_Chain', '').lower() == 'true':
                    typ = 'TCR'
                regions = literal(row.get('Region_Split_' + field), {})
                rec = dict(kind=typ, sequence=seq, cdr3=regions.get('cdr3', ''), cdr1=regions.get('cdr1', ''),
                           cdr2=regions.get('cdr2', ''), chain=row.get(old_id, ''), source=row['Name'])
                if rec not in PDB_ANNOTATIONS[row['PDB_ID'].upper()]:
                    PDB_ANNOTATIONS[row['PDB_ID'].upper()].append(rec)
    # SAbDab is a formal auxiliary VHH source, not only a resolution fallback.
    # Keep its chain-level metadata so formal ingestion can verify the VHH,
    # light-chain absence and antigen identity without borrowing SNAC labels.
    for path in sorted(root.rglob('*sabdab*summary*.tsv')):
        if _staging(root, path):
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            fields = set(reader.fieldnames or [])
            if not {'pdb', 'resolution'} <= fields:
                continue
            _record_source_file(root, path, 'sabdab_summary')
            for row in reader:
                pdb = str(row.get('pdb') or '').strip().upper()
                _record_resolution(pdb, row.get('resolution'), 'sabdab_summary')
                if re.fullmatch(r'[0-9A-Z]{4}', pdb) and {'Hchain', 'Lchain', 'antigen_chain'} <= fields:
                    if row not in SABDAB_ANNOTATIONS[pdb]:
                        SABDAB_ANNOTATIONS[pdb].append(row)
    for path in sorted(root.rglob('*entry_resolution.tsv')):
        if _staging(root, path):
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if not reader.fieldnames or not {'pdb', 'resolution'} <= set(reader.fieldnames):
                continue
            _record_source_file(root, path, 'rcsb_entry_resolution')
            for row in reader:
                _record_resolution(row.get('pdb'), row.get('resolution'), 'rcsb_entry_resolution')
    for path in root.rglob('all_input_PDB_files_parsed_file_chains.csv'):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                pid=row['PDB_ID'].upper()
                if row['Bioassembly']=='0' and pid in PDB_ANNOTATIONS:
                    CHAIN_ANNOTATIONS[pid]={k:literal(row.get(k),[]) for k in ('Chain_VH','Chain_VL','Chain_VHH')}

def task_source_sha256(task):
    """SHA-256 of the exact raw structure file or ZIP member audited."""
    cached = str(task.get('_source_structure_sha256') or '')
    if cached:
        return cached
    if task.get('member'):
        if not hasattr(ZIP_LOCAL, 'archives'):
            ZIP_LOCAL.archives = {}
        if task['path'] not in ZIP_LOCAL.archives:
            ZIP_LOCAL.archives[task['path']] = zipfile.ZipFile(task['path'])
        raw = ZIP_LOCAL.archives[task['path']].read(task['member'])
        digest = hashlib.sha256(raw).hexdigest()
    else:
        digest = sha256_file(pathlib.Path(task['path']))
    task['_source_structure_sha256'] = digest
    return digest

def structural(name):
    return name.lower().endswith(('.pdb', '.cif', '.cif.gz'))

def discover(root):
    tasks, ignored = [], []
    mapping = {'rcsb_non_redundant_dataset': 'train_rcsb', 'sabdab_all_sd_h_structures': 'sabdab_vhh', 'sabdab_all_single_domain_structures': 'sabdab_vhh', 'benchmark5.5': 'test_db55'}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or not structural(path.name):
            continue
        if path.name.startswith('._'):
            ignored.append(str(path.relative_to(root)))
            continue
        parts = path.relative_to(root).parts
        if parts[0] in PIPELINE_WORK_DIRS:
            continue
        subset = mapping.get(parts[0], parts[0] if parts[0] in ('train_rcsb', 'sabdab_vhh', 'snac_db', 'test_db55') else 'extra_' + parts[0])
        if parts[0] == 'SNAC-DataBase':
            subset = 'snac_db' if 'curated_structures' in parts and 'nb_complexes' in parts else 'extra_SNAC_loose'
        tasks.append(dict(path=str(path), member='', subset=subset, id=str(path.relative_to(root))))
    archives = []
    for path in sorted(root.rglob('*.zip')):
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if structural(n) and not pathlib.PurePosixPath(n).name.startswith('._')]
        selected = path.stem in ('nb_complexes', 'nb_unbound') and path.parent.name in ('curated_structures', 'benchmark_dataset')
        archives.append({'path': str(path.relative_to(root)), 'structures': len(members), 'audited': selected})
        if not selected:
            continue
        for member in members:
            # If already extracted, audit the physical file once, not twice.
            if (path.parent / member).exists():
                continue
            subset = 'snac_db' if path.parent.name == 'curated_structures' and path.stem == 'nb_complexes' else 'extra_snac_' + path.parent.name + '_' + path.stem
            tasks.append(dict(path=str(path), member=member, subset=subset, id=str(path.relative_to(root)) + '::' + member))
    return tasks, ignored, archives


def interresidue_heavy_overlap(chains, minimum_distance=None):
    """Count protein heavy-atom pairs on distinct residues below a hard floor."""
    floor=float(MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM if minimum_distance is None
                else minimum_distance)
    if not math.isfinite(floor) or not 0 < floor <= 1.0:
        raise ValueError('interresidue heavy-atom floor must be in (0, 1.0] A')
    xyz=[]; owners=[]; labels=[]
    for chain_index, chain in enumerate(chains):
        for atom_index, point in enumerate(chain['xyz']):
            residue_index=int(chain['owners'][atom_index])
            xyz.append(point)
            owners.append((chain_index,residue_index))
            labels.append(f"{chain['name']}:{chain['residue_seqids'][residue_index]}:"
                          f"{chain['residue_names'][residue_index]}:"
                          f"{chain['atom_names'][atom_index]}")
    if len(xyz)<2:
        return dict(count=0,minimum_distance_angstrom=None,closest_pair=None)
    coordinates=np.asarray(xyz,dtype=np.float64)
    pairs=cKDTree(coordinates).query_pairs(floor,output_type='ndarray')
    count=0; nearest=None; closest=None
    for i,j in pairs:
        if owners[i]==owners[j]:
            continue
        distance=float(np.linalg.norm(coordinates[i]-coordinates[j]))
        if distance>=floor:
            continue
        count+=1
        if nearest is None or distance<nearest:
            nearest=distance
            closest=[labels[i],labels[j]]
    return dict(count=count,minimum_distance_angstrom=nearest,closest_pair=closest)

def contact_residue_ids(a, b, cutoff=None):
    # Nearest-neighbour queries avoid enumerating every atom-atom pair.
    cutoff = float(INTERFACE_CONTACT_CUTOFF_ANGSTROM if cutoff is None else cutoff)
    query_bound=np.nextafter(cutoff,np.inf)
    da = b['tree'].query(a['xyz'], distance_upper_bound=query_bound)[0]
    db = a['tree'].query(b['xyz'], distance_upper_bound=query_bound)[0]
    return set(map(int,np.unique(a['owners'][da <= cutoff]))), set(map(int,np.unique(b['owners'][db <= cutoff])))

def contact(a, b):
    left,right=contact_residue_ids(a,b)
    return len(left),len(right)

def interfaces(chains, allowed=None):
    pairs = []
    for i, a in enumerate(chains):
        for b in chains[i+1:]:
            if allowed is not None and not allowed(a['name'], b['name']):
                continue
            if np.any(np.maximum(a['low']-b['high'], b['low']-a['high']) > INTERFACE_CONTACT_CUTOFF_ANGSTROM):
                na, nb = 0, 0
            else:
                na, nb = contact(a, b)
            pairs.append((a['name'], b['name'], na, nb))
    return pairs

def is_subsequence(short, long):
    it = iter(long)
    return all(c in it for c in short)

def _metadata_present(value):
    return str(value or '').strip().lower() not in ('', 'na', 'nan', 'none')


def _split_chain_ids(value):
    return {token.strip() for token in re.split(r'[|,]', str(value or ''))
            if _metadata_present(token)}


def _author_chain(name):
    """Author-chain identifier for a biological-assembly symmetry copy."""
    return str(name).split('-', 1)[0]


def sabdab_chain_metadata(pdbid):
    """Annotated antibody and antigen author-chain IDs for one SAbDab entry."""
    rows = SABDAB_ANNOTATIONS.get(str(pdbid).upper(), [])
    antibody = set()
    antigen = set()
    for row in rows:
        antibody.update(_split_chain_ids(row.get('Hchain')))
        antibody.update(_split_chain_ids(row.get('Lchain')))
        antigen.update(_split_chain_ids(row.get('antigen_chain')))
    return sorted(antibody), sorted(antigen)


def _annotate_sabdab_cdrs(sequence):
    from nanoqc.data.build_external_vhh_graphs import annotate_cdrs
    return annotate_cdrs(sequence, require_anarci=True)


def _ig_variable_domain(sequence):
    from nanoqc.data.build_foldseek_pairs import ig_variable_domain
    return ig_variable_domain(sequence)


def sabdab_features(chains, pdbid):
    """Strict SAbDab VHH identity from SAbDab metadata plus the structure."""
    result = dict(
        vhh_status='unknown', vhh_reason='缺少SAbDab链级标注',
        vhh_chain='', cdr1_sequences=[], cdr2_sequences=[], cdr3_lengths=[],
        cdr3_sequences=[], cdr_annotation_method='', sabdab_antigen_chains=[],
        other_antibody_chains=[],
    )
    rows = SABDAB_ANNOTATIONS.get(str(pdbid).upper(), [])
    if not rows:
        return result, None
    if any(_metadata_present(row.get('Lchain')) for row in rows):
        result.update(vhh_status='fail', vhh_reason='SAbDab标注含轻链/VH-VL')
        return result, None
    if any(str(row.get('scfv', '')).strip().lower() == 'true' for row in rows):
        result.update(vhh_status='fail', vhh_reason='SAbDab标注为scFv')
        return result, None
    vhh_ids = sorted({chain for row in rows for chain in _split_chain_ids(row.get('Hchain'))})
    if len(vhh_ids) != 1:
        result.update(vhh_status='fail',
                      vhh_reason=f'SAbDab VHH链不唯一（{len(vhh_ids)}条）')
        return result, None
    antigen_rows = [row for row in rows if _metadata_present(row.get('antigen_chain'))]
    antigen_ids = sorted({chain for row in antigen_rows
                          for chain in _split_chain_ids(row.get('antigen_chain'))})
    if not antigen_ids:
        result.update(vhh_status='fail', vhh_reason='SAbDab未标注抗原链')
        return result, None
    antigen_types = {token.strip().lower() for row in antigen_rows
                     for token in str(row.get('antigen_type', '')).split('|') if token.strip()}
    if not antigen_types.intersection({'protein', 'peptide'}):
        result.update(vhh_status='fail', vhh_reason='SAbDab无protein/peptide抗原')
        return result, None

    author_vhh = vhh_ids[0]
    heavy = [chain for chain in chains if chain['name'] == author_vhh]
    if not heavy:
        author_matches = [chain for chain in chains if _author_chain(chain['name']) == author_vhh]
        if len(author_matches) == 1:
            heavy = author_matches
    if len(heavy) != 1:
        result.update(vhh_status='fail', vhh_reason='SAbDab VHH链在生物学装配中缺失或不唯一')
        return result, None
    anchor = heavy[0]
    observed = anchor['sequence']
    if len(observed) < 70:
        result.update(vhh_status='fail', vhh_reason='SAbDab VHH坐标序列过短（<70 aa）')
        return result, None
    try:
        cdrs = _annotate_sabdab_cdrs(observed)
    except (ValueError, RuntimeError) as exc:
        result.update(vhh_status='unknown', vhh_reason=f'SAbDab CDR-H3无法确定：{exc}')
        return result, None

    extra_ig = []
    antigen_set = set(antigen_ids)
    for chain in chains:
        author = _author_chain(chain['name'])
        if chain['name'] == anchor['name'] or author in antigen_set or chain['sequence'] == observed:
            continue
        is_ig, method = _ig_variable_domain(chain['sequence'])
        if is_ig:
            extra_ig.append(dict(chain=chain['name'], method=method))
    if extra_ig:
        result.update(
            vhh_status='fail',
            vhh_reason='SAbDab结构含未标注的额外Ig可变域',
            other_antibody_chains=[entry['chain'] for entry in extra_ig],
        )
        return result, None

    result.update(
        vhh_status='pass',
        vhh_reason='SAbDab单VHH链级标注 + 生物学装配核对 + CDR-H3定位',
        vhh_chain=anchor['name'],
        cdr1_sequences=([cdrs['cdr1']] if cdrs.get('cdr1') else []),
        cdr2_sequences=([cdrs['cdr2']] if cdrs.get('cdr2') else []),
        cdr3_lengths=[len(cdrs['cdr3'])],
        cdr3_sequences=[cdrs['cdr3']],
        cdr_annotation_method=cdrs.get('method', ''),
        sabdab_antigen_chains=antigen_ids,
    )
    anchor_name = anchor['name']
    return result, lambda a, b: (
        (a == anchor_name and _author_chain(b) in antigen_set)
        or (b == anchor_name and _author_chain(a) in antigen_set)
    )


def nano_features(task, chains, pdbid):
    result = dict(
        vhh_status='unknown', vhh_reason='缺少可验证链标注',
        vhh_chain='', cdr1_sequences=[], cdr2_sequences=[], cdr3_lengths=[],
        cdr3_sequences=[], cdr_annotation_method='', sabdab_antigen_chains=[],
        other_antibody_chains=[],
    )
    if task.get('subset') == 'sabdab_vhh':
        return sabdab_features(chains, pdbid)

    name = pathlib.PurePosixPath(task['member'] or task['path']).stem
    parent = str(pathlib.Path(task['path']).parent)
    row = ANNOTATIONS.get((parent, name))
    if row is None and not task['member']:
        row = ANNOTATIONS.get((str(pathlib.Path(task['path']).parent.parent), name))
    if row is not None:
        if row.get('TCR_Chain', '').lower() == 'true':
            result.update(vhh_status='fail', vhh_reason='SNAC 标记 TCR')
            return result, None
        heavy = [c for c in chains if c['name'] == row.get('Chain_VH')]
        if row.get('Chain_VL', '').strip() not in ('', 'nan', 'None') or any(c['name'] == 'L' for c in chains):
            result.update(vhh_status='fail', vhh_reason='含轻链/VH-VL')
            return result, None
        if len(heavy) != 1:
            result.update(vhh_status='fail', vhh_reason='H 链缺失或不唯一')
            return result, None
        seq = row.get('Sequence_VH', '')
        obs = heavy[0]['sequence']
        if not seq or not is_subsequence(obs, seq) or len(obs) < .7 * len(seq):
            result.update(vhh_reason='H 链与标注序列不匹配或覆盖不足')
            return result, None
        regions = literal(row.get('Region_Split_VH'), {})
        cdr1, cdr2, cdr3 = regions.get('cdr1', ''), regions.get('cdr2', ''), regions.get('cdr3', '')
        if cdr3:
            result.update(
                cdr1_sequences=([cdr1] if cdr1 else []),
                cdr2_sequences=([cdr2] if cdr2 else []),
                cdr3_lengths=[len(cdr3)], cdr3_sequences=[cdr3],
                cdr_annotation_method='snac_imgt_region_split',
            )
        extensions=len(regions.get('c_st',''))+len(regions.get('c_e',''))
        if extensions>20:
            result.update(vhh_status='unknown', vhh_reason=f'可变域外端部扩展{extensions}aa，需复核是否融合域')
        else:
            result.update(vhh_status='pass', vhh_reason='SNAC 非TCR单VHH标注 + H链序列核对；端部扩展不超过20aa',
                          vhh_chain=heavy[0]['name'])
        antigen = set(literal(row.get('Chain_Ag'), []))
        return result, lambda a, b: (a == heavy[0]['name'] and b in antigen) or (b == heavy[0]['name'] and a in antigen)

    records = PDB_ANNOTATIONS.get(pdbid, [])
    matches = []
    for chain in chains:
        obs = chain['sequence']
        found = [r for r in records if len(obs) >= 70 and len(obs) >= .7 * len(r['sequence']) and is_subsequence(obs, r['sequence'])]
        kinds = {r['kind'] for r in found}
        if len(kinds) == 1:
            matches.append((chain['name'], found[0]['kind'], {r['cdr3'] for r in found if r['cdr3']}))
        elif kinds:
            result['vhh_reason'] = '跨标注链类型冲突'
            return result, None
    vhh = [m for m in matches if m[1] == 'VHH']
    other = [m for m in matches if m[1] != 'VHH']
    if other:
        result.update(vhh_status='fail', vhh_reason='检出VH/VL/TCR标注序列')
    elif len(vhh) > 1:
        result.update(vhh_status='fail', vhh_reason='含多个VHH链副本（非单VHH条目）')
    elif len(vhh) == 1:
        result.update(vhh_status='candidate', vhh_reason='一条VHH匹配；其他链未独立排除免疫球蛋白域',
                      vhh_chain=vhh[0][0])
        source=CHAIN_ANNOTATIONS.get(pdbid)
        if source:
            names={c['name'] for c in chains}
            domains=lambda key:[x for x in source[key] if x.rsplit('_',1)[0] in names]
            hh,hl,hv=domains('Chain_VHH'),domains('Chain_VL'),domains('Chain_VH')
            if len(hh)==1 and not hl and set(hv)==set(hh) and hh[0].rsplit('_',1)[0]==vhh[0][0]:
                result.update(vhh_status='pass',vhh_reason='ASU0全链标注仅单VHH域 + 非TCR序列匹配')
            elif hl or len(hv)>1 or len(hh)>1:
                result.update(vhh_status='fail',vhh_reason='ASU0标注含额外VH/VL域')
    if vhh:
        unique = {next(iter(m[2])) for m in vhh if len(m[2]) == 1}
        if len(unique) == 1 and all(len(m[2]) == 1 for m in vhh):
            cdr = next(iter(unique))
            result.update(cdr3_lengths=[len(cdr)], cdr3_sequences=[cdr],
                          cdr_annotation_method='snac_transferred_annotation')
        elif len(unique) > 1:
            result['vhh_reason'] += '；多种CDR-H3，文件级长度未判定'
        vhhnames = {m[0] for m in vhh}
        return result, lambda a, b: (a in vhhnames) != (b in vhhnames)
    return result, None


def formal_source_by_pdb(rows):
    """Highest-priority source represented for each PDB, before QC/outcomes."""
    rank = {source: index for index, source in enumerate(FORMAL_SOURCE_PRIORITY)}
    represented = collections.defaultdict(set)
    for row in rows:
        pdb = str(row.get('pdb_id') or '').strip().upper()
        subset = str(row.get('subset') or '')
        if pdb and subset in rank:
            represented[pdb].add(subset)
    return {
        pdb: min(sources, key=lambda source: rank[source])
        for pdb, sources in represented.items()
    }


def formal_row_eligible(row, min_interface_residues=15):
    """The row-level QC gate used before formal graph construction/clustering."""
    if row.get('subset') not in FORMAL_VHH_SUBSETS:
        return False
    if row.get('valid') is not True:
        return False
    if int(row.get('missing_residues') or 0) > 0:
        return False
    if row.get('interface_status') != 'pass':
        return False
    maximum = row.get('max_contact_residues')
    if maximum is None or int(maximum) < int(min_interface_residues):
        return False
    if row.get('vhh_status') != 'pass':
        return False
    if row.get('structure_quality_status') != 'pass':
        return False
    return True


def formal_clustering_pdb_ids(rows, min_interface_residues=15):
    """PDB universe matching source precedence and formal row-level admission.

    Lower-priority representations never rescue a preferred source that fails
    QC. Excluded structures therefore cannot bridge otherwise independent
    Foldseek components.
    """
    preferred = formal_source_by_pdb(rows)
    out = set()
    for pdb, source in preferred.items():
        if source not in FORMAL_VHH_SUBSETS:
            continue
        if any(str(row.get('pdb_id') or '').strip().upper() == pdb
               and row.get('subset') == source
               and formal_row_eligible(row, min_interface_residues)
               for row in rows):
            out.add(pdb.lower())
    return sorted(out)


def audit(task):
    out = dict(task, valid=False, error='', residues=0, missing_residues=0, missing_examples=[], chains=0,
        interface_status='not_applicable', max_contact_residues=None, weak_pairs=0, pairs=[],
        models_first_only=False, vhh_status='not_applicable', cdr3_lengths=[],
        source_structure_sha256='',
        resolution_angstrom=None, resolution_source=None, structure_quality_status='not_evaluated',
        structure_quality_reasons=[], interface_missing_sidechain_residues=0,
        interface_altloc_residues=0, interface_min_occupancy=None,
        interface_mean_bfactor=None, interresidue_heavy_overlap_count=0,
        interresidue_heavy_min_distance_angstrom=None,
        interresidue_heavy_closest_pair=None)
    try:
        st, multi = read_structure(task)
        if task.get('subset') in FORMAL_VHH_SUBSETS or task.get('subset') == 'test_db55':
            out['source_structure_sha256'] = task_source_sha256(task)
        if not len(st):
            raise ValueError('no coordinate model')
        chains, missing, total, details = chain_data(st[0])
        if not chains or not total:
            raise ValueError('no usable amino-acid heavy atoms')
        name = pathlib.Path(task['member'] or task['path']).name
        pdbid = re.search(r'pdb_0000([a-zA-Z0-9]{4})', name)
        pdbid = pdbid.group(1).upper() if pdbid else name[:4].upper()
        resolution=float(getattr(st,'resolution',0.0) or 0.0)
        resolution_source='structure_file'
        if not math.isfinite(resolution) or resolution<=0:
            resolution,resolution_source=RESOLUTION_BY_PDB.get(pdbid,(None,None))
        out.update(valid=True, pdb_id=pdbid, chains=len(chains), residues=total,
            missing_residues=missing, missing_examples=details[:20],
            models_first_only=multi or len(st)>1,
            legacy_pdb_tail=task.get('_legacy_pdb_tail',False),
            resolution_angstrom=resolution, resolution_source=resolution_source)
        allowed = None
        if task['subset'] in ('sabdab_vhh', 'snac_db') or task['subset'].startswith('extra_snac_'):
            features, allowed = nano_features(task, chains, pdbid)
            out.update(features)
        # DB5.5 component files are not independently docking complexes.
        if task['subset'] == 'test_db55':
            return out
        pairs = interfaces(chains, allowed)
        out['pairs'] = pairs
        if pairs:
            maximum = max(p[2]+p[3] for p in pairs)
            out.update(max_contact_residues=maximum, weak_pairs=sum(p[2]+p[3]<15 for p in pairs), interface_status='weak' if maximum < 15 else 'pass')
            best=max(pairs,key=lambda p:p[2]+p[3])
            cmap={chain['name']:chain for chain in chains}
            left,right=cmap[best[0]],cmap[best[1]]
            overlap=interresidue_heavy_overlap([left,right])
            out.update(interresidue_heavy_overlap_count=overlap['count'],
                       interresidue_heavy_min_distance_angstrom=overlap['minimum_distance_angstrom'],
                       interresidue_heavy_closest_pair=overlap['closest_pair'])
            left_ids,right_ids=contact_residue_ids(left,right)
            interface_records=[]
            for chain,ids in ((left,left_ids),(right,right_ids)):
                for rid in ids:
                    interface_records.append(dict(
                        chain=chain['name'],rid=int(rid),name=chain['residue_names'][rid],
                        missing_sidechain=chain['residue_missing_sidechain'][rid],
                        altloc=bool(chain['residue_altloc'][rid]),
                        min_occupancy=float(chain['residue_min_occ'][rid]),
                        mean_bfactor=float(chain['residue_mean_b'][rid]),
                    ))
            missing_sc=sum(bool(r['missing_sidechain']) for r in interface_records)
            altloc_sc=sum(bool(r['altloc']) for r in interface_records)
            min_occ=min((r['min_occupancy'] for r in interface_records),default=None)
            bvals=[r['mean_bfactor'] for r in interface_records if math.isfinite(r['mean_bfactor'])]
            reasons=[]
            if REQUIRE_RESOLUTION and resolution is None:
                reasons.append('unknown_resolution')
            if resolution is not None and resolution>MAX_RESOLUTION_ANGSTROM:
                reasons.append('resolution_above_limit')
            if REQUIRE_COMPLETE_INTERFACE_SIDECHAINS and missing_sc:
                reasons.append('incomplete_interface_sidechain')
            if not ALLOW_INTERFACE_ALTLOC and altloc_sc:
                reasons.append('interface_altloc')
            if min_occ is not None and min_occ<MIN_INTERFACE_OCCUPANCY:
                reasons.append('low_interface_occupancy')
            if overlap['count']:
                reasons.append('nonphysical_interresidue_heavy_overlap')
            out.update(
                interface_missing_sidechain_residues=missing_sc,
                interface_altloc_residues=altloc_sc,
                interface_min_occupancy=min_occ,
                interface_mean_bfactor=(float(np.mean(bvals)) if bvals else None),
                interface_quality_records=interface_records,
                structure_quality_status=('pass' if not reasons else 'fail'),
                structure_quality_reasons=reasons,
            )
        return out
    except Exception as exc:
        raise_if_resource_error(exc, stage_hint="data audit")
        out.update(valid=False, error=f'{type(exc).__name__}: {exc}')
        return out

def db55_pairs(tasks, pool=None):
    """Pair rows in discovery order; ``pool`` computes the contacts concurrently (A36)."""
    bypath = {t['path']: t for t in tasks if t['subset']=='test_db55'}
    results = []
    pending = []
    for path, task in bypath.items():
        if not path.endswith('_r_b.pdb'):
            continue
        partner = path[:-8] + '_l_b.pdb'
        out = dict(id=pathlib.Path(path).name[:4], receptor=path, ligand=partner, valid=False, error='missing bound ligand')
        results.append(out)
        if partner in bypath:
            pending.append((out, (task, bypath[partner])))
    contacts = (list(pool.map(_db55_pair_contacts, [pair for _, pair in pending]))
                if pool is not None and pending
                else [_db55_pair_contacts(pair) for _, pair in pending])
    for (out, _pair), (nr, nl, error) in zip(pending, contacts):
        if error:
            out['error']=error
            continue
        out.update(valid=True,error='', receptor_contacts=nr,ligand_contacts=nl,contact_residues=nr+nl,interface_status='weak' if nr+nl<15 else 'pass')
    return results

def pct(n,d,digits=1):
    return f'{n/d:.{digits}%}' if d else 'N/A'

def report(root, rows, ignored, archives, pairs, elapsed, destination):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['subset']].append(r)
    lines=['# 全数据集结构快速审计报告','',f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}。',f'数据目录：`{root}`。Gemmi {gemmi.__version__}；NumPy {np.__version__}。','', '## 统计口径与范围','',
    '- 有效文件：格式可解析，首模型含至少一个具有正占有率、有限坐标的氨基酸重原子残基；不等于完整结构或独立样本。忽略 `._` AppleDouble 资源文件。',
    '- 主链缺失：已观测氨基酸残基缺少 N、CA、C、O 任一原子；同名原子选最高占有率构象。完全未建模的残基不在分母内，本次不根据 SEQRES 补计。非蛋白链、水和游离配体不参与。',
    f'- VHH主界面：跨伙伴重原子严格距离 <{INTERFACE_CONTACT_CUTOFF_ANGSTROM:.1f} Å；统计两侧接触残基数之和。多链结构默认取最强链对；有VHH标注时仅比较VHH–已标注抗原链。DB5.5辅助配对继续使用CAPRI式5.0 Å接触定义。',
    '- 合格率是严格基础筛查率：有效、观测残基主链无缺失、可评估界面且接触≥15；分母为有效文件。纳米专区的“VHH严格通过率”另列，不混同基础物理合格率。',
    '- DB5.5 的单独 receptor/ligand 文件只做格式与主链检查；界面按 `_r_b`+`_l_b` 结合态坐标配对统计，不对未结合态强行叠合。',
    '- 快速模式只计算每个文件首模型。CAPRI 多模型 PDB 仅读至首个 ENDMDL，后续模型既未解析也未做完整性验证；本报告不是全部 decoy 的质量分布。',
    '- SNAC 主表 snac_db = curated_structures/nb_complexes（ZIP 内直接读）；nb_unbound 和 benchmark/nb_complexes 单列。其他 SNAC ZIP 只登记、不解析内部结构；所有松散 .pdb/.cif/.cif.gz 均已纳入。',
    '- CDR-H3：SNAC 使用自身 IMGT Region_Split_VH.cdr3；SAbDab 使用自身 Hchain 元数据定位VHH后进行 IMGT/保守锚点 CDR-H3 定位，并逐条记录 cdr_annotation_method。分箱为 <12、12–15、≥16，避免16 aa重复计数。',
    '- 分辨率：优先取结构文件自带值；文件未记录时按 PDB ID 回落到整理元数据（SNAC curation summary 的 Resolution 列、SAbDab 汇总表的 resolution 列），逐行记录 resolution_source。一个字段列出多个值时取最差（最大）值。流水线工作目录（external_vhh）下的文件不参与，本报告列出实际使用的元数据文件及其哈希。仍然查不到分辨率的条目按 unknown_resolution 排除，阈值不变。',
    '- VHH通过 = SNAC非TCR单VHH来源标注、唯一H链、无L链、H链序列覆盖≥70%且与标注一致；或SAbDab链级summary明确唯一Hchain、无Lchain/scFv、具有protein/peptide抗原，并在生物学装配中核对VHH链、CDR-H3和额外Ig可变域。SNAC与SAbDab标注独立读取，SAbDab不得借用同PDB的SNAC标签。未知/候选不得当作合格。','',
    '## 核心子集规模与基础质量','', '| 子集 | 结构文件 | 有效 | 解析/坐标失败 | 有主链缺失文件（占有效） | 缺原子残基/观测残基 | 可评估界面 | 弱界面 | 基础合格/有效 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    order=['train_rcsb','sabdab_vhh','snac_db','test_db55']
    def summary(g):
        rr=groups[g];v=[r for r in rr if r['valid']];m=sum(r['missing_residues']>0 for r in v);nr=sum(r['residues'] for r in v);nm=sum(r['missing_residues'] for r in v); ev=sum(r['interface_status']!='not_applicable' for r in v);w=sum(r['interface_status']=='weak' for r in v);good=sum(r['missing_residues']==0 and r['interface_status']=='pass' for r in v)
        quality=f'{good}/{len(v)} ({pct(good,len(v))})' if g!='test_db55' else '见配对统计'
        if g.endswith('nb_unbound'):quality='N/A（未结合单链）'
        return f'| {g} | {len(rr)} | {len(v)} | {len(rr)-len(v)} | {m} ({pct(m,len(v))}) | {nm}/{nr} ({pct(nm,nr,3)}) | {ev} | {w} | {quality} |'
    lines += [summary(g) for g in order]
    lines += ['', '## 非物理重原子重叠筛查', '',
              f'- 门槛：首模型最强许可蛋白链对的不同残基重原子中心距 <{MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM:g} Å；同残基原子对不计。该绝对下限只筛明显坐标重叠，不是 MolProbity clashscore，也不筛生成的氢或候选 rotamer。',
              '| 子集 | 有效且可评估界面 | 距离门槛排除 | 重叠原子对 |',
              '|---|---:|---:|---:|']
    for g in ['snac_db','sabdab_vhh']:
        eligible=[r for r in groups[g] if r['valid'] and r.get('pairs')]
        excluded=[r for r in eligible if int(r.get('interresidue_heavy_overlap_count') or 0)>0]
        lines.append(f'| {g} | {len(eligible)} | {len(excluded)} | '
                     f'{sum(int(r.get("interresidue_heavy_overlap_count") or 0) for r in eligible)} |')
    lines += ['', '逐结构最近原子对、距离和计数见 `data_audit_details.csv` / `data_audit_details.jsonl`。', '']
    validpairs=[p for p in pairs if p['valid']];weakpairs=[p for p in validpairs if p['interface_status']=='weak']
    lookup={r['path']:r for r in rows if r['subset']=='test_db55'}
    paired_good=sum(p['interface_status']=='pass' and all(lookup.get(p[k],{}).get('valid') and lookup[p[k]]['missing_residues']==0 for k in ('receptor','ligand')) for p in validpairs)
    lines += ['',f'DB5.5：发现 {len(pairs)} 个结合态配对，{len(validpairs)} 个成功评估，{len(weakpairs)} 个界面接触残基<15；界面通过率 {pct(len(validpairs)-len(weakpairs),len(validpairs))}。同时满足两侧观测残基主链完整、界面通过的配对为 {paired_good}/{len(validpairs)}（基础合格率 {pct(paired_good,len(validpairs))}）。', '']
    # After the core table, never between its header and its rows.
    if RESOLUTION_SOURCE_FILES:
        lines += ['## 分辨率元数据来源','', '| 文件 | 类型 | SHA-256 |','|---|---|---|']
        lines += [f'| `{f["path"]}` | {f["kind"]} | `{f["sha256"]}` |' for f in RESOLUTION_SOURCE_FILES]
        lines += ['']
    lines += ['## 纳米抗体身份与 CDR-H3','', '| 子集 | 有效 | VHH严格通过 | 单VHH候选 | 不满足单VHH条件 | 未判定 | 严格通过率 |','|---|---:|---:|---:|---:|---:|---:|']
    for g in ['sabdab_vhh','snac_db']:
        v=[r for r in groups[g] if r['valid']];c=collections.Counter(r['vhh_status'] for r in v)
        lines.append(f'| {g} | {len(v)} | {c["pass"]} | {c["candidate"]} | {c["fail"]} | {c["unknown"]} | {pct(c["pass"],len(v))} |')
    lines += ['', '| 子集/口径 | 可判定长度文件 | <12 aa | 12–15 aa | ≥16 aa | 长度未判定 |','|---|---:|---:|---:|---:|---:|']
    for g in ['sabdab_vhh','snac_db']:
        for strict in [False,True]:
            v=[r for r in groups[g] if r['valid'] and (not strict or r['vhh_status']=='pass')];lens=[r['cdr3_lengths'][0] for r in v if len(r['cdr3_lengths'])==1];bins=[sum(x<12 for x in lens),sum(12<=x<16 for x in lens),sum(x>=16 for x in lens)]
            lines.append(f'| {g}/'+('严格VHH' if strict else '全部可注释文件')+f' | {len(lens)} | '+' | '.join(f'{n} ({pct(n,len(lens))})' for n in bins)+f' | {len(v)-len(lens)} |')
    lines += ['', '百分比分母为该行可判定长度的文件数；相同CDR的多链副本在文件级只计一次，不同CDR并存则不强行赋予单一长度。','', '## 其余松散结构与补充子集','', '| 子集 | 结构文件 | 有效 | 解析/坐标失败 | 有主链缺失文件（占有效） | 缺原子残基/观测残基 | 可评估界面 | 弱界面 | 基础合格/有效 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines += [summary(g) for g in sorted(groups) if g not in order]
    lines += ['', f'忽略 AppleDouble 资源文件 {len(ignored)} 个；首模型审计且检测到 MODEL 标记/多个模型的文件 {sum(r["models_first_only"] for r in rows)} 个。',f'兼容读取旧版PDB尾部字段的文件 {sum(r.get("legacy_pdb_tail",False) for r in rows)} 个：仅在电荷字段解析失败时忽略第67列及之后的旧编号，保留坐标/占有率/B因子，由原子名推断元素；不修改源文件。','', '## 压缩包覆盖范围','', '| 压缩包 | 内部结构文件 | 本次解析 |','|---|---:|---|']
    lines += [f'| {a["path"]} | {a["structures"]} | {"是（或已解压文件去重）" if a["audited"] else "否，仅登记"} |' for a in archives]
    lines += ['', '## 格式破损、无有效蛋白坐标或疑似弱界面清单','', '完整清单如下；主链缺失和VHH不符合条件的逐条记录另见 `data_audit_details.csv` / `data_audit_details.jsonl`。','', '| 子集 | PDB/条目 | 异常 | 文件/ZIP成员 |','|---|---|---|---|']
    anomalies=0
    for r in rows:
        if not r['valid'] or r['interface_status']=='weak':
            msg=r['error'] if not r['valid'] else f'最强界面仅{r["max_contact_residues"]}个接触残基'
            lines.append(f'| {r["subset"]} | {r.get("pdb_id", "未知")} | {msg.replace(chr(124),"/")} | `{r["id"]}` |');anomalies+=1
    for p in pairs:
        if not p['valid'] or p.get('interface_status')=='weak':
            lines.append(f'| test_db55 配对 | {p["id"]} | {p["error"] or str(p["contact_residues"])+"个接触残基"} | `{p["receptor"]}` + `{p["ligand"]}` |');anomalies+=1
    if not anomalies:lines.append('| — | — | 无 | — |')
    lines += ['', '## 可复现性与限制','', '- 逐文件结果：`data_audit_details.csv`、`data_audit_details.jsonl`；DB5.5配对：`data_audit_db55_pairs.json`；范围清单：`data_audit_inventory.json`。', '- 本次不去重、不划分训练集/测试集、不判断跨库泄漏、不生成生物学装配、不做能量松弛；目录名train/test仅为用户指定用途映射。', '- 原子缺失和小界面阈值属于初筛，不等同实验结构质量、亲和力或生物学真实性。单链/无合适抗原链为界面不适用，不标成伪复合物。', '- 解析实现参考：[Gemmi 官方接口](https://project-gemmi.github.io/python-api/gemmi.html)；SNAC链命名、TCR标识、IMGT分区依据本地README和CSV标注。']
    destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')

# The annotation tables every worker needs, in the order annotation_snapshot
# and _initialize_audit_process pass them.
ANNOTATION_TABLES = ('ANNOTATIONS', 'PDB_ANNOTATIONS', 'CHAIN_ANNOTATIONS',
                     'SABDAB_ANNOTATIONS', 'RESOLUTION_BY_PDB', 'RESOLUTION_SOURCE_FILES')


def annotation_snapshot():
    """The parent's loaded annotation tables, to hand to workers (A36).

    Re-reading them per worker meant one recursive walk of the data root and one
    parse of every summary file per process. Passing the loaded tables gives
    every worker the same snapshot by construction.
    """
    return tuple(globals()[name] for name in ANNOTATION_TABLES)


def install_annotations(annotations):
    """Install an :func:`annotation_snapshot` in this process (A36).

    Any worker pool whose tasks read the annotation tables uses this as its
    initializer instead of ``load_annotations``, which would walk the whole data
    root and re-parse every summary file once per process.
    """
    for name, table in zip(ANNOTATION_TABLES, annotations):
        target = globals()[name]
        if target is table:
            continue  # already installed; clearing first would empty the source
        # Copy before clearing: a snapshot taken in this same process can alias
        # a nested container of the target.
        source = list(table) if isinstance(target, list) else dict(table)
        target.clear()
        if isinstance(target, list):
            target.extend(source)
        else:
            target.update(source)


def _initialize_audit_process(annotations, thresholds):
    """Install the parent's immutable annotation and threshold snapshot."""
    global INTERFACE_CONTACT_CUTOFF_ANGSTROM, MAX_RESOLUTION_ANGSTROM
    global MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM, MIN_INTERFACE_OCCUPANCY
    global ALLOW_INTERFACE_ALTLOC, REQUIRE_RESOLUTION, REQUIRE_COMPLETE_INTERFACE_SIDECHAINS
    install_annotations(annotations)
    (INTERFACE_CONTACT_CUTOFF_ANGSTROM, MAX_RESOLUTION_ANGSTROM,
     MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM, MIN_INTERFACE_OCCUPANCY,
     ALLOW_INTERFACE_ALTLOC, REQUIRE_RESOLUTION,
     REQUIRE_COMPLETE_INTERFACE_SIDECHAINS) = thresholds


def _db55_pair_contacts(pair):
    """``(receptor_contacts, ligand_contacts, error)`` for one DB5.5 pair.

    Failures are returned, not raised, so one bad pair cannot discard the other
    pairs' results when this runs in a worker pool (A36); the recorded error text
    is the same one the serial code stored.
    """
    try:
        return (*_db55_contact_counts(pair), '')
    except Exception as exc:
        raise_if_resource_error(exc, stage_hint="DB5.5 contacts")
        return None, None, str(exc)


def _db55_contact_counts(pair):
    task, partner_task = pair
    a, _, _, _ = chain_data(read_structure(task)[0][0])
    b, _, _, _ = chain_data(read_structure(partner_task)[0][0])
    if not a or not b:
        raise ValueError('empty receptor/ligand')
    # Union per residue over all receptor-ligand chain pairs.
    ac = dict(xyz=np.concatenate([c['xyz'] for c in a]), owners=np.concatenate([c['owners']+sum(len(x['sequence']) for x in a[:i]) for i,c in enumerate(a)]))
    bc = dict(xyz=np.concatenate([c['xyz'] for c in b]), owners=np.concatenate([c['owners']+sum(len(x['sequence']) for x in b[:i]) for i,c in enumerate(b)]))
    ac['tree'], bc['tree'] = cKDTree(ac['xyz']), cKDTree(bc['xyz'])
    # DB5.5/CAPRI auxiliary semantics remain the benchmark's conventional 5 A
    # atom-contact definition, independent of the VHH primary 4.5 A label.
    left_ids, right_ids = contact_residue_ids(ac, bc, cutoff=5.0)
    return len(left_ids), len(right_ids)


def main():
    global INTERFACE_CONTACT_CUTOFF_ANGSTROM, MAX_RESOLUTION_ANGSTROM, MIN_INTERFACE_OCCUPANCY
    global MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM
    global ALLOW_INTERFACE_ALTLOC, REQUIRE_RESOLUTION, REQUIRE_COMPLETE_INTERFACE_SIDECHAINS
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=pathlib.Path,default=BASE/'data');parser.add_argument('--workers',type=int,default=4);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--out',type=pathlib.Path,default=BASE);parser.add_argument('--interface-contact-cutoff',type=float,default=INTERFACE_CONTACT_CUTOFF_ANGSTROM);parser.add_argument('--max-resolution',type=float,default=MAX_RESOLUTION_ANGSTROM);parser.add_argument('--min-interresidue-heavy-distance',type=float,default=MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM);parser.add_argument('--min-interface-occupancy',type=float,default=MIN_INTERFACE_OCCUPANCY);parser.add_argument('--allow-interface-altloc',action='store_true');parser.add_argument('--allow-unknown-resolution',action='store_true');parser.add_argument('--allow-incomplete-interface-sidechains',action='store_true');parser.add_argument('--reuse-non-nano',action='store_true',help='Explicitly reuse valid non-nano geometry from this output directory; assumes unchanged files and geometry rules. Nano annotations and failed entries are recomputed.');args=parser.parse_args()
    if not math.isfinite(args.interface_contact_cutoff) or args.interface_contact_cutoff<=0: parser.error('--interface-contact-cutoff must be positive finite')
    if not math.isfinite(args.max_resolution) or args.max_resolution<=0: parser.error('--max-resolution must be positive finite')
    if not math.isfinite(args.min_interresidue_heavy_distance) or not 0<args.min_interresidue_heavy_distance<=1.0: parser.error('--min-interresidue-heavy-distance must be in (0,1.0] A')
    if not 0 < args.min_interface_occupancy <= 1: parser.error('--min-interface-occupancy must be in (0,1]')
    INTERFACE_CONTACT_CUTOFF_ANGSTROM=float(args.interface_contact_cutoff)
    MAX_RESOLUTION_ANGSTROM=float(args.max_resolution)
    MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM=float(args.min_interresidue_heavy_distance)
    MIN_INTERFACE_OCCUPANCY=float(args.min_interface_occupancy)
    ALLOW_INTERFACE_ALTLOC=bool(args.allow_interface_altloc)
    REQUIRE_RESOLUTION=not bool(args.allow_unknown_resolution)
    REQUIRE_COMPLETE_INTERFACE_SIDECHAINS=not bool(args.allow_incomplete_interface_sidechains)
    start=time.time();root=args.data.resolve();args.out.mkdir(parents=True,exist_ok=True);load_annotations(root);tasks,ignored,archives=discover(root)
    if args.limit:tasks=tasks[:args.limit]
    print(f'Found {len(tasks)} structures; ignored {len(ignored)} resource files',flush=True)
    cached={}
    previous=args.out/'data_audit_details.jsonl'
    if args.reuse_non_nano and previous.exists():
        stamp=previous.stat().st_mtime
        for r in iter_jsonl(previous,skip_undecodable=True):
            if r['valid'] and r['subset'] not in ('sabdab_vhh','snac_db') and not r['subset'].startswith('extra_snac_') and pathlib.Path(r['path']).exists() and pathlib.Path(r['path']).stat().st_mtime<=stamp:
                cached[r['id']]=r
        print(f'Reusing {len(cached)} valid non-nano geometry records; rechecking all nano annotations and failures',flush=True)
    rows=[]
    thresholds=(INTERFACE_CONTACT_CUTOFF_ANGSTROM,MAX_RESOLUTION_ANGSTROM,
                MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM,MIN_INTERFACE_OCCUPANCY,
                ALLOW_INTERFACE_ALTLOC,REQUIRE_RESOLUTION,REQUIRE_COMPLETE_INTERFACE_SIDECHAINS)
    if args.workers < 1: parser.error('--workers must be positive')
    # Cached rows are merged here, not sent to a worker and back, and the
    # DB5.5 pair contacts reuse this pool instead of running serially (A36).
    fresh=[task for task in tasks if task['id'] not in cached]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers,max(1,len(fresh)+1)),mp_context=mp.get_context('spawn'),
            initializer=_initialize_audit_process,
            initargs=(annotation_snapshot(),thresholds)) as pool, \
            (args.out/'data_audit_details.jsonl').open('w',encoding='utf-8') as handle:
        audited=iter(pool.map(audit,fresh))
        for task in tasks:
            r=cached.get(task['id']) or next(audited)
            rows.append(r);handle.write(json.dumps(r,ensure_ascii=False)+'\n')
            if len(rows)%250==0:handle.flush();print(f'{len(rows)}/{len(tasks)} audited, {time.time()-start:.0f}s',flush=True)
        pairs=db55_pairs(tasks,pool)
    fields=['subset','id','pdb_id','valid','error','chains','residues','missing_residues','interface_status','max_contact_residues','weak_pairs','vhh_status','vhh_reason','vhh_chain','cdr_annotation_method','cdr3_lengths','sabdab_antigen_chains','other_antibody_chains','source_structure_sha256','models_first_only','legacy_pdb_tail','resolution_angstrom','resolution_source','structure_quality_status','structure_quality_reasons','interface_missing_sidechain_residues','interface_altloc_residues','interface_min_occupancy','interface_mean_bfactor','interresidue_heavy_overlap_count','interresidue_heavy_min_distance_angstrom','interresidue_heavy_closest_pair']
    with (args.out/'data_audit_details.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (args.out/'data_audit_db55_pairs.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out/'data_audit_inventory.json').write_text(json.dumps(dict(
        ignored=ignored,archives=archives,tasks=len(tasks),partial_run=bool(args.limit),
        execution=dict(mode='spawned_processes',workers=args.workers,ordered_output=True),
        source_hash_contract=dict(
            algorithm='sha256',
            bound_subsets=sorted(FORMAL_VHH_SUBSETS | {'test_db55'}),
            hashed_valid_rows=sum(bool(r.get('source_structure_sha256')) for r in rows),
        ),
        structure_quality_protocol=dict(interface_contact_cutoff_angstrom=INTERFACE_CONTACT_CUTOFF_ANGSTROM,
            max_resolution_angstrom=MAX_RESOLUTION_ANGSTROM,
            min_interresidue_heavy_distance_angstrom=MIN_INTERRESIDUE_HEAVY_DISTANCE_ANGSTROM,
            min_interface_occupancy=MIN_INTERFACE_OCCUPANCY,
            allow_interface_altloc=ALLOW_INTERFACE_ALTLOC,
            require_resolution=REQUIRE_RESOLUTION,
            require_complete_interface_sidechains=REQUIRE_COMPLETE_INTERFACE_SIDECHAINS),
        interresidue_heavy_overlap_audit=dict(
            scope='strongest_allowed_protein_chain_pair_in_first_model',
            evaluated_rows=sum(bool(r.get('valid') and r.get('pairs')) for r in rows),
            excluded_rows=sum(bool(r.get('valid') and r.get('interresidue_heavy_overlap_count',0)>0)
                              for r in rows),
            overlap_atom_pairs=sum(int(r.get('interresidue_heavy_overlap_count') or 0) for r in rows))
    ),ensure_ascii=False,indent=2),encoding='utf-8')
    report(root,rows,ignored,archives,pairs,time.time()-start,args.out/'data_audit_report.md')
    print(f'Done: {args.out / "data_audit_report.md"}',flush=True)

if __name__=='__main__':
    main()
