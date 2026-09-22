"""Build audited PyG residue graphs and a deterministic CDR-H3-disjoint split.

Run with the project's .venv Python. All processing is CPU-only.
Dependencies: requirements-graph.txt and requirements-audit.txt.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import csv
import functools
import hashlib
import itertools
import json
import math
import pathlib
import random
import threading
import time
import traceback

import gemmi
import numpy as np
import parasail
import psutil
import torch
import torch_geometric
from scipy.spatial import cKDTree
from torch_geometric.data import Data, Batch
import audit_all_datasets as audit

BASE = pathlib.Path(__file__).resolve().parent
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {a:i for i,a in enumerate(AA)}
SEED = 20260917
CLUSTER_IDENTITY_THRESHOLD = 0.40
INTERFACE_LABEL_CUTOFF_ANGSTROM = 5.0
INTRA_CHAIN_CA_CUTOFF_ANGSTROM = 8.0
CROSS_PARTNER_KNN_K = 3
MIN_INTERFACE_RESIDUES = 15
VERSION = '1.3'
PROCESS = psutil.Process()
PEAK_RSS = 0
MEMORY_LOCK = threading.Lock()
RESUME = False

def memory_sample():
    global PEAK_RSS
    with MEMORY_LOCK:
        PEAK_RSS=max(PEAK_RSS, PROCESS.memory_info().rss)

def sha256(path):
    with path.open('rb') as f:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(f, 'sha256').hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()

def write_csv(path, records, fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)

@functools.lru_cache(maxsize=300000)
def similarity(a,b):
    """Symmetric global identity, exact matches / alignment length incl. gaps."""
    if a==b:return 1.0
    if not a or not b or min(len(a),len(b))/max(len(a),len(b)) < CLUSTER_IDENTITY_THRESHOLD:return 0.0
    r=parasail.nw_stats_striped_16(a,b,10,1,parasail.blosum62)
    s=parasail.nw_stats_striped_16(b,a,10,1,parasail.blosum62)
    if r.saturated or s.saturated:raise ValueError('alignment score saturation')
    return max(r.matches/r.length,s.matches/s.length)

def seqsim(a,b):
    return similarity(*sorted((a,b)))

def cdr(row):
    seqs=row.get('cdr3_sequences',[])
    return seqs[0] if len(seqs)==1 else ''

def audit_reasons(row):
    reasons=[]
    if not row['valid']:reasons.append('audit_invalid')
    if row['missing_residues']>0:reasons.append('missing_backbone')
    if row['interface_status']=='weak' or (row.get('max_contact_residues') is not None and row['max_contact_residues']<MIN_INTERFACE_RESIDUES):reasons.append('weak_interface')
    if row['interface_status']!='pass' and 'weak_interface' not in reasons:reasons.append('no_eligible_interface')
    if row['subset'] in ('sabdab_vhh','snac_db') and row['vhh_status']!='pass':reasons.append('not_strict_vhh')
    return reasons

def load_inputs(root):
    paths=[root/n for n in ['data_audit_report.md','data_audit_details.csv','data_audit_details.jsonl','data_audit_db55_pairs.json']]
    report=paths[0].read_text(encoding='utf-8')
    if '核心子集' not in report:raise ValueError('Unexpected audit report')
    table={r['id']:r for r in csv.DictReader(paths[1].open(encoding='utf-8-sig',newline=''))}
    rows=[json.loads(line) for line in paths[2].open(encoding='utf-8')]
    for r in rows:
        q=table[r['id']]
        for key in ['subset','valid','missing_residues','interface_status','vhh_status']:
            if str(r[key]) != q[key]:raise ValueError(f'CSV/JSONL audit mismatch: {r["id"]} {key}')
    if len(table)!=len(rows):raise ValueError('Audit row count mismatch')
    pairs=json.loads(paths[3].read_text(encoding='utf-8'))
    return rows,pairs,{p.name:sha256(p) for p in paths}

def cluster_long(rows):
    """Cluster long CDR-H3 sequences by 40%-identity connected components."""
    byseq=collections.defaultdict(list)
    for r in rows:
        if len(cdr(r))>=16:
            byseq[cdr(r)].append(r)

    sequences=sorted(byseq,key=lambda s:(-len(s),s))
    parent=list(range(len(sequences)))

    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i

    def union(i,j):
        ri,rj=find(i),find(j)
        if ri!=rj:
            parent[rj]=ri

    for i,seq_i in enumerate(sequences):
        for j in range(i):
            seq_j=sequences[j]
            if min(len(seq_i),len(seq_j))/max(len(seq_i),len(seq_j)) < CLUSTER_IDENTITY_THRESHOLD:
                continue
            if seqsim(seq_i,seq_j)>=CLUSTER_IDENTITY_THRESHOLD:
                union(i,j)

    components=collections.defaultdict(list)
    for i,seq in enumerate(sequences):
        components[find(i)].append(seq)

    clusters=[]
    ordered_components=sorted(
        components.values(),
        key=lambda seqs:(-max(len(s) for s in seqs),min(seqs)),
    )
    for i,seqs in enumerate(ordered_components):
        representative=sorted(seqs,key=lambda s:(-len(s),s))[0]
        members=[row for seq in seqs for row in byseq[seq]]
        cl=dict(
            representative=representative,
            members=members,
            sequences=sorted(seqs),
            cluster_id=f'cdr40_{i:04d}',
        )
        cl['candidates']=sorted(
            [r for r in members if cdr(r)==representative],
            key=lambda r:(-r['max_contact_residues'],r['id']),
        )
        clusters.append(cl)
    return clusters

def build_atoms(st, prefix='', identity_overrides=None):
    """Full protein residue nodes; recheck every retained heavy-atom residue."""
    if not len(st):raise ValueError('no model')
    chains=[]
    for chain in st[0]:
        nodes=[];heavy=[];owners=[];seen=set()
        for res in chain:
            info=gemmi.find_tabulated_residue(res.name)
            if not info.is_amino_acid() or res.entity_type in (gemmi.EntityType.Water,gemmi.EntityType.NonPolymer):continue
            atoms={}
            for atom in res:
                if atom.occ<=0 or atom.element.is_hydrogen:continue
                xyz=(atom.pos.x,atom.pos.y,atom.pos.z)
                if not all(math.isfinite(v) for v in xyz):raise ValueError('non-finite coordinate')
                if atom.name not in atoms or atom.occ>atoms[atom.name].occ:atoms[atom.name]=atom
            if not atoms:continue
            if audit.BACKBONE-atoms.keys():raise ValueError(f'missing backbone on reread: {chain.name}:{res.seqid}')
            aa=(identity_overrides or {}).get((chain.name,str(res.seqid)),info.one_letter_code.upper())
            if aa not in AA_INDEX:raise ValueError(f'unsupported amino acid {res.name}/{aa}; cannot encode 20-way one-hot')
            key=str(res.seqid)
            if key in seen:raise ValueError(f'ambiguous duplicate residue ID: {chain.name}:{key}')
            seen.add(key)
            ca=atoms['CA'].pos
            nodes.append(dict(aa=aa,pos=(ca.x,ca.y,ca.z),residue_id=prefix+chain.name+':'+key,name=res.name))
            for atom in atoms.values():
                heavy.append((atom.pos.x,atom.pos.y,atom.pos.z));owners.append(len(nodes)-1)
        if nodes:chains.append(dict(name=prefix+chain.name,original_name=chain.name,nodes=nodes,xyz=np.array(heavy),owners=np.array(owners,dtype=np.int64)))
    if not chains:raise ValueError('no amino acid nodes')
    if len({c['name'] for c in chains})!=len(chains):raise ValueError('ambiguous duplicate chain IDs')
    return chains

def task_of(row):
    return {k:row[k] for k in ['path','member','subset','id']}

def bound_identity_overrides(st,row):
    """Resolve ambiguous labels only from the explicitly paired DB5.5 apo file.

    Same residue numbers, >=80% coverage and no standard-residue disagreement
    are required; no coordinates are copied or filled.
    """
    ambiguous=[(c.name,str(r.seqid),gemmi.find_tabulated_residue(r.name).one_letter_code.upper()) for c in st[0] for r in c if gemmi.find_tabulated_residue(r.name).is_amino_acid() and gemmi.find_tabulated_residue(r.name).one_letter_code.upper() not in AA_INDEX]
    if not ambiguous:return {},[]
    apo=pathlib.Path(row['path'].replace('_b.pdb','_u.pdb'))
    if not apo.exists():return {},[]
    task=dict(task_of(row),path=str(apo));reference=audit.read_structure(task)[0]
    proposals=collections.defaultdict(set)
    for chain in st[0]:
        bound={str(r.seqid):gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in chain if gemmi.find_tabulated_residue(r.name).is_amino_acid()}
        for other in reference[0]:
            ref={str(r.seqid):gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in other if gemmi.find_tabulated_residue(r.name).is_amino_acid()}
            common=set(bound)&set(ref)
            if len(common)<.8*len(bound) or not common:continue
            if any(bound[k] in AA_INDEX and bound[k]!=ref[k] for k in common):continue
            for key in common:
                if bound[key] in ('Z','B') and ref[key] in ({'E','Q'} if bound[key]=='Z' else {'D','N'}):proposals[(chain.name,key)].add(ref[key])
    overrides={k:next(iter(v)) for k,v in proposals.items() if len(v)==1}
    notes=[dict(chain=c,residue_id=r,from_code=code,to_code=overrides[(c,r)],evidence=str(apo)) for c,r,code in ambiguous if (c,r) in overrides]
    return overrides,notes

def extract(row, pair=None):
    if pair:
        rec=audit.read_structure(task_of(pair['receptor_row']))[0];lig=audit.read_structure(task_of(pair['ligand_row']))[0]
        ro,rn=bound_identity_overrides(rec,pair['receptor_row']);lo,ln=bound_identity_overrides(lig,pair['ligand_row'])
        a=build_atoms(rec,'R:',ro);b=build_atoms(lig,'L:',lo)
        for chain in a:chain['identity_resolutions']=rn
        for chain in b:chain['identity_resolutions']=ln
        for c in a:c['group']=0
        for c in b:c['group']=1
        return a+b
    st,_=audit.read_structure(task_of(row));chains=build_atoms(st)
    if row['subset']=='snac_db':anchor='H'
    elif row['subset']=='sabdab_vhh':
        source=audit.CHAIN_ANNOTATIONS.get(row['pdb_id'].upper(),{})
        names={c['name'] for c in chains}
        candidates=[d.rsplit('_',1)[0] for d in source.get('Chain_VHH',[]) if d.rsplit('_',1)[0] in names]
        if len(candidates)!=1:raise ValueError('strict VHH anchor not unique on reread')
        anchor=candidates[0]
    else:
        if not row['pairs']:raise ValueError('missing audited contact pairs')
        strongest=sorted(row['pairs'],key=lambda p:(-(p[2]+p[3]),p[0],p[1]))[0]
        anchor=strongest[0]
    if sum(c['name']==anchor for c in chains)!=1:raise ValueError('anchor chain absent or ambiguous')
    for c in chains:c['group']=0 if c['name']==anchor else 1
    if {c['group'] for c in chains}!={0,1}:raise ValueError('both interaction partners required')
    return chains

def make_graph(row, split, pair=None):
    chains=extract(row,pair);memory_sample()
    nodes=[];groups=[];chainidx=[];heavy={0:[],1:[]};owners={0:[],1:[]}
    for i,c in enumerate(chains):
        offset=len(nodes);nodes.extend(c['nodes']);groups.extend([c['group']]*len(c['nodes']));chainidx.extend([i]*len(c['nodes']))
        heavy[c['group']].append(c['xyz']);owners[c['group']].append(c['owners']+offset)
    n=len(nodes)
    if n>200000:raise MemoryError(f'node safety cap exceeded: {n}')
    arrays={g:np.concatenate(heavy[g]) for g in [0,1]};rid={g:np.concatenate(owners[g]) for g in [0,1]}
    trees={g:cKDTree(arrays[g]) for g in [0,1]}
    counts=[]
    interface_nodes=set()
    for g in [0,1]:
        distances=trees[1-g].query(arrays[g],distance_upper_bound=INTERFACE_LABEL_CUTOFF_ANGSTROM)[0]
        contacted=rid[g][distances<INTERFACE_LABEL_CUTOFF_ANGSTROM]
        unique_contacted=np.unique(contacted)
        counts.append(len(unique_contacted))
        interface_nodes.update(int(index) for index in unique_contacted.tolist())
    interface=sum(counts)
    if interface<MIN_INTERFACE_RESIDUES:raise ValueError(f'weak actual partner interface: {interface}')
    heavy_atom_interface_label=np.zeros(n,dtype=np.float32)
    if interface_nodes:
        heavy_atom_interface_label[np.fromiter(sorted(interface_nodes),dtype=np.int64)]=1.0
    del arrays,rid,trees,heavy,owners
    # Leakage control: target labels are defined from cross-partner heavy-atom
    # contacts (configured heavy-atom cutoff), so cross-partner edge EXISTENCE must not be defined by a
    # similar distance threshold.  Keep local same-chain CA-radius edges, then
    # connect every residue to a fixed number of nearest residues on the other
    # partner.  This preserves antigen context without making "has a cross edge"
    # an almost-direct proxy for the interface label.
    pos=np.array([r['pos'] for r in nodes],dtype=np.float32)
    group_array=np.asarray(groups,dtype=np.int64);chain_array=np.asarray(chainidx,dtype=np.int64)
    pair_set=set()
    for chain_index in sorted(set(chainidx)):
        global_indices=np.flatnonzero(chain_array==chain_index)
        if len(global_indices)<2:continue
        local_pos=pos[global_indices]
        local_pairs=cKDTree(local_pos).query_pairs(INTRA_CHAIN_CA_CUTOFF_ANGSTROM,output_type='ndarray')
        if len(local_pairs):
            candidate=global_indices[local_pairs]
            delta=pos[candidate[:,0]].astype(np.float64)-pos[candidate[:,1]].astype(np.float64)
            candidate=candidate[np.einsum('ij,ij->i',delta,delta)<INTRA_CHAIN_CA_CUTOFF_ANGSTROM**2]
            pair_set.update(tuple(sorted(map(int,p))) for p in candidate)
    cross_partner_knn_k=CROSS_PARTNER_KNN_K
    for g in [0,1]:
        source=np.flatnonzero(group_array==g);partner=np.flatnonzero(group_array==1-g)
        if not len(source) or not len(partner):raise ValueError('both partner groups required for cross-partner KNN')
        k=min(cross_partner_knn_k,len(partner))
        _,nearest=cKDTree(pos[partner]).query(pos[source],k=k)
        nearest=np.asarray(nearest)
        if nearest.ndim==1:nearest=nearest[:,None]
        for row_index,node in enumerate(source):
            for partner_local in nearest[row_index]:
                pair_set.add(tuple(sorted((int(node),int(partner[int(partner_local)])))))
    pairs=np.asarray(sorted(pair_set),dtype=np.int64)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not len(pairs):raise ValueError('edge construction produced no pairs')
    budget=min(4*1024**3,int(psutil.virtual_memory().available*.25))
    if len(pairs)>10000000 or len(pairs)*128+n*512>budget:raise MemoryError(f'edge allocation budget exceeded: {len(pairs)} undirected edges')
    edge=np.concatenate([pairs.T,pairs[:,::-1].T],axis=1).astype(np.int64,copy=False)
    x=np.zeros((n,21),dtype=np.float32);x[np.arange(n),[AA_INDEX[r['aa']] for r in nodes]]=1;x[:,20]=groups
    seq=cdr(row)
    chain_sequences=[''.join(r['aa'] for r in c['nodes']) for c in chains]
    vhh_sequences=[s for s,c in zip(chain_sequences,chains) if c['group']==0]
    antigen_sequences=[s for s,c in zip(chain_sequences,chains) if c['group']==1]
    graph=Data(pos=torch.from_numpy(pos),x=torch.from_numpy(x),edge_index=torch.from_numpy(edge),
        interface_label=torch.from_numpy(heavy_atom_interface_label),
        pdb_id=row['pdb_id'].upper(),subset_source=row['subset'],cdr3_seq=seq,cdr3_len=len(seq),num_interface_residues=interface,
        split=split,source_id=row['id'],node_chain_id=torch.tensor(chainidx,dtype=torch.long),chain_ids=[c['name'] for c in chains],
        chain_groups=[c['group'] for c in chains],residue_ids=[r['residue_id'] for r in nodes],
        chain_sequences=chain_sequences,vhh_sequences=vhh_sequences,antigen_sequences=antigen_sequences,
        edge_policy='intra_chain_ca_radius_plus_cross_partner_knn',
        intra_chain_ca_cutoff_angstrom=float(INTRA_CHAIN_CA_CUTOFF_ANGSTROM),
        cross_partner_knn_k=cross_partner_knn_k,
        label_policy='cross_partner_heavy_atom_cutoff',
        interface_label_cutoff_angstrom=float(INTERFACE_LABEL_CUTOFF_ANGSTROM),
        min_interface_residues=int(MIN_INTERFACE_RESIDUES),
        audit_interface_residues=row.get('max_contact_residues',pair['contact_residues'] if pair else 0),
        audit_vhh_status=row.get('vhh_status','not_applicable'),graph_version=VERSION)
    gnotes={json.dumps(note,sort_keys=True) for chain in chains for note in chain.get('identity_resolutions',[])}
    graph.residue_identity_resolutions=json.dumps([json.loads(s) for s in sorted(gnotes)])
    graph.num_nodes=n
    validate_graph(graph)
    memory_sample()
    return graph

def validate_graph(g):
    n=g.num_nodes;e=g.edge_index
    assert isinstance(g,Data) and g.pos.shape==(n,3) and g.pos.dtype==torch.float32
    assert g.x.shape==(n,21) and g.x.dtype==torch.float32 and e.dtype==torch.long and e.shape[0]==2
    assert torch.isfinite(g.pos).all() and torch.isfinite(g.x).all()
    assert torch.all(g.x[:,:20].sum(1)==1) and torch.all((g.x==0)|(g.x==1))
    assert set(g.x[:,20].tolist())=={0.0,1.0}
    assert e.shape[1]%2==0 and e.shape[1]>0 and e.min()>=0 and e.max()<n
    m=e.shape[1]//2
    assert torch.equal(e[:,:m].flip(0),e[:,m:]) and torch.all(e[0]!=e[1])
    first=e[:,:m].numpy();codes=first[0]*n+first[1]
    assert len(np.unique(codes))==m
    pos=g.pos.numpy().astype(np.float64);d=pos[first[0]]-pos[first[1]];d2=np.einsum('ij,ij->i',d,d)
    chain=g.node_chain_id.numpy();group=g.x[:,-1].numpy()
    same_chain=chain[first[0]]==chain[first[1]]
    cross_partner=group[first[0]]!=group[first[1]]
    assert np.all(same_chain|cross_partner)
    ca_cutoff=float(getattr(g,'intra_chain_ca_cutoff_angstrom',0.0))
    label_cutoff=float(getattr(g,'interface_label_cutoff_angstrom',0.0))
    min_interface=int(getattr(g,'min_interface_residues',0))
    assert ca_cutoff>0 and label_cutoff>0 and min_interface>0
    assert np.all(d2[same_chain]<ca_cutoff**2)
    assert getattr(g,'edge_policy','')=='intra_chain_ca_radius_plus_cross_partner_knn'
    assert getattr(g,'label_policy','')=='cross_partner_heavy_atom_cutoff'
    assert int(getattr(g,'cross_partner_knn_k',0))>0
    for node in range(n):
        mask=(first[0]==node)|(first[1]==node)
        assert np.any(mask & cross_partner), f'node {node} lacks threshold-independent cross-partner context'
    assert len(getattr(g,'vhh_sequences',[]))>=1 and len(getattr(g,'antigen_sequences',[]))>=1
    assert g.num_interface_residues>=min_interface and len(g.cdr3_seq)==g.cdr3_len
    assert g.validate(raise_on_error=True)

def save_graph(row, split, output, pair=None, cluster_id=''):
    try:
        name=f'{row["subset"]}__{row["pdb_id"].upper()}__{hashlib.sha256(row["id"].encode()).hexdigest()[:16]}.pt'
        path=output/'graphs'/split/name
        existing=path.exists()
        if RESUME and existing:
            g=torch.load(path,map_location='cpu',weights_only=False);validate_graph(g)
            if g.source_id!=row['id'] or g.split!=split or g.graph_version!=VERSION:raise ValueError('Resume metadata mismatch')
            if not hasattr(g,'residue_identity_resolutions'):
                g.residue_identity_resolutions='[]'
                torch.save(g,path)
        else:
            g=make_graph(row,split,pair)
            torch.save(g,path)
        # Read-back validation ensures these are actual loadable PyG Data objects.
        loaded=torch.load(path,map_location='cpu',weights_only=False);validate_graph(loaded)
        if not torch.equal(loaded.pos,g.pos) or not torch.equal(loaded.edge_index,g.edge_index):raise ValueError('serialization mismatch')
        record=dict(split=split,path=str(path.relative_to(output)),source_id=row['id'],pdb_id=g.pdb_id,subset_source=row['subset'],nodes=g.num_nodes,
            directed_edges=g.num_edges,undirected_edges=g.num_edges//2,density=g.num_edges/(g.num_nodes*(g.num_nodes-1)),bytes=path.stat().st_size,
            cdr3_seq=g.cdr3_seq,cdr3_len=g.cdr3_len,num_interface_residues=g.num_interface_residues,cluster_id=cluster_id,sha256=sha256(path),identity_resolution_notes=getattr(g,'residue_identity_resolutions','[]'))
        return record,None
    except Exception as exc:
        # Only remove an incomplete file created by this call, inside the output tree.
        if 'path' in locals() and not locals().get('existing',False) and path.exists() and output.resolve() in path.resolve().parents:path.unlink()
        return None,dict(split=split,source_id=row['id'],pdb_id=row.get('pdb_id',''),reason=type(exc).__name__,detail=str(exc))

def exclusion(row,reasons):
    return dict(source_id=row['id'],pdb_id=row.get('pdb_id',''),subset_source=row['subset'],reasons=';'.join(reasons))

def delivery_report(output,manifest,exclusions,failures,summary,complete):
    totalbytes=sum(r['bytes'] for r in manifest)
    lines=['# PyG 图数据集交付报告','',f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}；状态：'+('完成并通过交付检查。' if complete else '未完成；请查看异常与运行摘要。'),'',
    '## 1. 流水分流统计','', '| 分流 | 有效 .pt | 总节点 | 无向边（唯一） | edge_index列数（双向） |','|---|---:|---:|---:|---:|']
    for split in ['train','test_db55','test_snac_hard']:
        rr=[r for r in manifest if r['split']==split]
        lines.append(f'| {split} | {len(rr)} | {sum(r["nodes"] for r in rr):,} | {sum(r["undirected_edges"] for r in rr):,} | {sum(r["directed_edges"] for r in rr):,} |')
    lines += ['', '### 按来源拆分','', '| 分流 / 原始来源 | 图数 |','|---|---:|']
    for (sp,src),count in sorted(collections.Counter((r['split'],r['subset_source']) for r in manifest).items()):lines.append(f'| {sp} / {src} | {count} |')
    lines += ['', '## 2. 拓扑与尺度分布','', '| 分流 | 平均节点 | 节点中位数 / P95 / 最大 | 平均无向边 | 平均双向边列数 | 平均图密度 |','|---|---:|---:|---:|---:|---:|']
    for split in ['train','test_db55','test_snac_hard']:
        rr=[r for r in manifest if r['split']==split]
        if not rr:continue
        n=np.array([r['nodes'] for r in rr]);lines.append(f'| {split} | {n.mean():.2f} | {np.median(n):.0f} / {np.percentile(n,95):.0f} / {n.max()} | {np.mean([r["undirected_edges"] for r in rr]):.2f} | {np.mean([r["directed_edges"] for r in rr]):.2f} | {np.mean([r["density"] for r in rr]):.6f} |')
    lines += ['', '图密度按每图 `2E/[N(N−1)]` 计算后取算术平均；E为无向唯一边数。edge_index同时存储(i,j)和(j,i)，不含自环。','',
    '## 3. CDR-H3 实际落位','', '| 分流 / 来源 | <12 aa | 12–15 aa | ≥16 aa | 未标注/不适用 |','|---|---:|---:|---:|---:|']
    for sp,src in sorted({(r['split'],r['subset_source']) for r in manifest}):
        rr=[r for r in manifest if r['split']==sp and r['subset_source']==src];lengths=[r['cdr3_len'] for r in rr]
        lines.append(f'| {sp} / {src} | {sum(0<x<12 for x in lengths)} | {sum(12<=x<16 for x in lengths)} | {sum(x>=16 for x in lengths)} | {sum(x==0 for x in lengths)} |')
    lines += ['', '## 4. 清洗、切分与去冗余','', '| 来源 | 审计候选 | 同时通过硬过滤 |','|---|---:|---:|']
    for src,counts in summary.get('admission',{}).items():lines.append(f'| {src} | {counts["input"]} | {counts["eligible"]} |')
    lines += ['',f'- DB5.5目标：248个主链完整且界面通过的bound受体–配体对；实际交付 {sum(r["split"]=="test_db55" for r in manifest)}。',
        f'- SNAC长CDR-H3：{summary.get("long_eligible",0)}条非DB5.5重叠候选，{summary.get("unique_long_cdr",0)}条唯一序列，{CLUSTER_IDENTITY_THRESHOLD*100:.0f}%代表簇 {summary.get("clusters",0)} 个；固定随机种子 {SEED} 选取目标400个，实际 {sum(r["split"]=="test_snac_hard" for r in manifest)}。',
        '- 去冗余口径由用户确认：仅CDR-H3；全局Needleman–Wunsch、BLOSUM62、gap-open=10、gap-extend=1，相同残基数/含gap的比对长度达到配置的identity阈值时才进入精确相似性判定。取正反向比对身份率较大值，避免最优比对并列导致方向差异。',
        f'- 贪心按CDR长度降序、序列字典序选代表，代表间身份率均<{CLUSTER_IDENTITY_THRESHOLD:.2f}；每簇仅一个代表进入挑战集。代表同序列多个结构优先选审计接触数较大的条目。固定种子打乱簇顺序，构图失败时尝试同代表序列的其他结构，再补选其他簇。',
        f'- 用户确认隔离泄漏：训练集排除两组测试的同PDB ID条目；排除CDR-H3与挑战集任一代表身份率≥{CLUSTER_IDENTITY_THRESHOLD:.2f}的条目。RCSB也保守检查本地同PDB的已知VHH CDR标注。被隔离的簇成员不回流训练集。',
        '- 硬过滤和隔离的数量可能重叠；逐样本多原因记录见 `excluded_samples.csv`。SAbDab 847是身份通过数，叠加物理条件后为708，不按847强行入库。',
        '- 本切分保证已核验的PDB与CDR层面隔离，不声称抗原家族、全长VHH同源性或未知免疫链的完全独立。未标注的RCSB隐含VHH仍需更深入序列注释排查。',
        '', '| 排除原因（可重叠） | 条目数 |','|---|---:|']
    for reason,count in sorted(collections.Counter(x for r in exclusions for x in r['reasons'].split(';')).items()):lines.append(f'| {reason} | {count} |')
    lines += ['', '## 5. 张量规范与链定义','',
        f'- `pos`: CA坐标float32[N,3]；`x`: float32[N,21]，氨基酸列顺序 `{AA}`，末列为相互作用组0/1。',
        '- 全部蛋白链和残基保留。通用多链复合物以最强界面链对中的首链为组0、其余蛋白链为组1；VHH为0、其他蛋白链为1；DB5.5受体链集合为0、配体链集合为1。0/1是伙伴组，不冒充多条物理链的唯一编号。',
        '- `chain_ids`、`node_chain_id`、`residue_ids`、`chain_sequences`保留真实链/残基信息；node_chain_id为图内局部链序号，DB5.5用R:/L:前缀防止链ID冲突。',
        f'- `edge_index`: int64[2,2E]；同链边使用CA距离<{INTRA_CHAIN_CA_CUTOFF_ANGSTROM:.2f}Å，跨伙伴边固定采用每节点{CROSS_PARTNER_KNN_K}个最近邻（不设接触距离阈值）。界面标签独立由跨伙伴重原子<{INTERFACE_LABEL_CUTOFF_ANGSTROM:.2f}Å定义，因此跨链边的“存在/不存在”不复用标签阈值。',
        f'- `num_interface_residues`重新计算两个伙伴组之间重原子距离<{INTERFACE_LABEL_CUTOFF_ANGSTROM:.2f}Å的两侧接触残基并集大小；准入下限为{MIN_INTERFACE_RESIDUES}。原审计值保存在audit_interface_residues。',
        '- 属性包括pdb_id、subset_source、cdr3_seq、cdr3_len及num_interface_residues。未标注CDR时使用空字符串与长度0，便于批处理。',
        '- 首模型、正占有率原子、同名原子取最高占有率；完整重读并复查N/CA/C/O，不修补缺失结构。Gemmi可映射到标准字母的修饰残基按标准氨基酸编码；不能映射到20字母的残基导致整条样本跳过，不使用全零伪one-hot。',
        '- DB5.5唯一允许的歧义标注核对：利用官方配套apo文件，要求相同残基编号覆盖≥80%、所有明确残基无冲突，GLX仅可解析为E/Q、ASX仅可解析为D/N且结论唯一。只补充节点类别，不替换坐标/原子。逐图证据见identity_resolution_notes和Data.residue_identity_resolutions。',
        '- 严格VHH沿用审计通过口径（来源域标注+序列核验，非独立ANARCI分类）。其他SNAC类别、CAPRI和审计未覆盖的压缩数据不参与本流水线。','',
        '## 6. 存储与硬件开销','',f'- 图文件实际总大小：{totalbytes/1e6:.2f} MB / {totalbytes/1e9:.3f} GB（十进制）；{totalbytes/1024**3:.3f} GiB。',
        f'- 进程采样峰值RSS：{PEAK_RSS/1024**3:.3f} GiB；全部CPU构建，未使用CUDA，构建显存占用为0。RSS为采样值，不是操作系统保证的绝对峰值。',
        '- 单图保护上限：200,000节点、20,000,000条双向边；分配估算超过min(4GiB,当时可用内存25%)则记录跳过。该保护不裁剪图。',
        f'- 总执行时间：{summary.get("elapsed_seconds",0):.1f}秒；torch {torch.__version__}、PyG {torch_geometric.__version__}、Gemmi {gemmi.__version__}、parasail {parasail.__version__}。','',
        '## 7. 验证与异常清单','',
        f'- 每个.pt保存后重新加载并校验Data类型、张量形状/类型、one-hot、坐标有限性、边索引范围、双向对称、无自环/重复边、同链严格{INTRA_CHAIN_CA_CUTOFF_ANGSTROM:.2f}Å阈值、跨伙伴固定KNN策略及≥{MIN_INTERFACE_RESIDUES}界面准入。',
        f'- 最终检查测试CDR代表两两<{CLUSTER_IDENTITY_THRESHOLD*100:.0f}%、训练/测试PDB互斥、已知训练CDR与挑战CDR<{CLUSTER_IDENTITY_THRESHOLD*100:.0f}%、图文件数量与清单一致，并实测PyG Batch批处理。',
        f'- 解析、编码或内存异常跳过共 {len(failures)} 条；不含正常的审计过滤及泄漏隔离。',
        '', '| 分流 | PDB | 条目 | 类型 | 说明 |','|---|---|---|---|---|']
    for r in failures:lines.append(f'| {r.get("split", "运行")} | {r.get("pdb_id", "")} | {r.get("source_id", "")} | {r["reason"]} | {r["detail"].replace(chr(124),"/").replace(chr(10)," ")} |')
    if not failures:lines.append('| — | — | — | — | 无 |')
    lines += ['', '## 8. 交付文件与使用','',
        '- `graphs/train/`、`graphs/test_db55/`、`graphs/test_snac_hard/`：PyG Data文件。',
        '- `graph_manifest.csv` / `.json`：逐图节点、边、密度、大小、来源、CDR与SHA256；`excluded_samples.csv`：正常过滤/隔离；`processing_failures.csv`：异常；`cdr3_clusters.json`：全部簇及代表；`run_summary.json`：参数、输入摘要与检查结果。',
        '- 源数据不修改；已有.pt默认拒绝混入旧结果。`--resume`仅在审计输入哈希一致时重用当前输出，并逐图重新验证；改变输入应使用新的`--out`路径。',
        '', '```python','import torch','from torch_geometric.loader import DataLoader','from pathlib import Path',
        '# 仅对自己生成、可信的本地文件使用 weights_only=False',
        'paths = sorted(Path("dataset_clean/graphs/train").glob("*.pt"))',
        'graph = torch.load(paths[0], map_location="cpu", weights_only=False)',
        'print(graph.pos.shape, graph.x.shape, graph.edge_index.shape)',
        '```','',
        '方法接口参考：[PyG Data与无向边示例](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)、[parasail全局比对](https://github.com/jeffdaily/parasail-python)。']
    (output/'graph_dataset_delivery_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    global RESUME, PEAK_RSS, CLUSTER_IDENTITY_THRESHOLD
    global INTERFACE_LABEL_CUTOFF_ANGSTROM, INTRA_CHAIN_CA_CUTOFF_ANGSTROM
    global CROSS_PARTNER_KNN_K, MIN_INTERFACE_RESIDUES
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=pathlib.Path,default=BASE/'dataset_clean_500');parser.add_argument('--workers',type=int,default=2);parser.add_argument('--target-hard',type=int,default=500);parser.add_argument('--no-cap',action='store_true',help='Process every qualifying, deduplicated, isolated CDR-H3 cluster instead of capping at --target-hard.');parser.add_argument('--partition-seed',type=int,default=None,help='Override the module SEED for cluster shuffle order (e.g. an independently-derived partition stream); defaults to SEED when omitted.');parser.add_argument('--audit-dir',type=pathlib.Path,default=BASE);parser.add_argument('--data-root',type=pathlib.Path,default=BASE/'data');parser.add_argument('--identity-threshold',type=float,default=CLUSTER_IDENTITY_THRESHOLD);parser.add_argument('--interface-label-cutoff',type=float,default=INTERFACE_LABEL_CUTOFF_ANGSTROM);parser.add_argument('--intra-chain-ca-cutoff',type=float,default=INTRA_CHAIN_CA_CUTOFF_ANGSTROM);parser.add_argument('--cross-partner-knn-k',type=int,default=CROSS_PARTNER_KNN_K);parser.add_argument('--min-interface-residues',type=int,default=MIN_INTERFACE_RESIDUES);parser.add_argument('--resume',action='store_true');args=parser.parse_args();RESUME=args.resume
    if not (0.0 < args.identity_threshold < 1.0): parser.error('--identity-threshold must be in (0,1)')
    if not (math.isfinite(args.interface_label_cutoff) and args.interface_label_cutoff > 0): parser.error('--interface-label-cutoff must be positive finite')
    if not (math.isfinite(args.intra_chain_ca_cutoff) and args.intra_chain_ca_cutoff > 0): parser.error('--intra-chain-ca-cutoff must be positive finite')
    if args.cross_partner_knn_k < 1: parser.error('--cross-partner-knn-k must be >=1')
    if args.min_interface_residues < 1: parser.error('--min-interface-residues must be >=1')
    CLUSTER_IDENTITY_THRESHOLD=float(args.identity_threshold)
    INTERFACE_LABEL_CUTOFF_ANGSTROM=float(args.interface_label_cutoff)
    INTRA_CHAIN_CA_CUTOFF_ANGSTROM=float(args.intra_chain_ca_cutoff)
    CROSS_PARTNER_KNN_K=int(args.cross_partner_knn_k)
    MIN_INTERFACE_RESIDUES=int(args.min_interface_residues)
    if not args.no_cap and not 300<=args.target_hard<=500:parser.error('--target-hard must be between 300 and 500 (or pass --no-cap to remove the cap entirely)')
    output=args.out.resolve();output.mkdir(parents=True,exist_ok=True)
    if RESUME:
        prior=json.loads((output/'run_summary.json').read_text(encoding='utf-8'))
        if prior.get('no_cap',False)!=args.no_cap or (not args.no_cap and prior.get('target_hard')!=args.target_hard):
            raise ValueError('Resume target differs or is unknown; choose a fresh output directory')
        expected_protocol=dict(identity_threshold=CLUSTER_IDENTITY_THRESHOLD,
            interface_label_cutoff_angstrom=INTERFACE_LABEL_CUTOFF_ANGSTROM,
            intra_chain_ca_cutoff_angstrom=INTRA_CHAIN_CA_CUTOFF_ANGSTROM,
            cross_partner_knn_k=CROSS_PARTNER_KNN_K,
            min_interface_residues=MIN_INTERFACE_RESIDUES,graph_version=VERSION)
        if prior.get('graph_protocol')!=expected_protocol:
            raise ValueError('Resume graph protocol differs; choose a fresh output directory')
    if any((output/'graphs').rglob('*.pt')) and not RESUME:raise FileExistsError('Output already contains graphs; choose a fresh --out directory or explicit --resume')
    for split in ['train','test_db55','test_snac_hard']:(output/'graphs'/split).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);start=time.time();manifest=[];exclusions=[];failures=[];summary={};complete=False;previous_elapsed=0
    try:
        rows,pairs,input_hashes=load_inputs(args.audit_dir);audit.load_annotations(args.data_root)
        if RESUME:
            previous=json.loads((output/'run_summary.json').read_text(encoding='utf-8'))
            if previous.get('input_sha256')!=input_hashes:raise ValueError('Resume audit input hashes differ')
            previous_elapsed=previous.get('elapsed_seconds',0)
            PEAK_RSS=max(PEAK_RSS,previous.get('sampled_peak_rss_bytes',0))
        partition_seed=args.partition_seed if args.partition_seed is not None else SEED
        summary.update(seed=partition_seed,no_cap=args.no_cap,target_hard=(None if args.no_cap else args.target_hard),identity_threshold=CLUSTER_IDENTITY_THRESHOLD,identity_scope='CDR-H3 hard-set isolation; EGNN train/validation uses bilateral full-chain VHH+antigen clustering',graph_protocol=dict(identity_threshold=CLUSTER_IDENTITY_THRESHOLD,interface_label_cutoff_angstrom=INTERFACE_LABEL_CUTOFF_ANGSTROM,intra_chain_ca_cutoff_angstrom=INTRA_CHAIN_CA_CUTOFF_ANGSTROM,cross_partner_knn_k=CROSS_PARTNER_KNN_K,min_interface_residues=MIN_INTERFACE_RESIDUES,graph_version=VERSION),input_sha256=input_hashes,script_sha256=sha256(pathlib.Path(__file__)),admission={})
        lookup={r['path']:r for r in rows if r['subset']=='test_db55'};eligible=[]
        for source in ['train_rcsb','sabdab_vhh','snac_db']:
            subset=[r for r in rows if r['subset']==source];good=[]
            for r in subset:
                reasons=audit_reasons(r)
                if reasons:exclusions.append(exclusion(r,reasons))
                else:good.append(r)
            summary['admission'][source]=dict(input=len(subset),eligible=len(good));eligible.extend(good)
        bound=[]
        for p in pairs:
            source=dict(id='DB55_BOUND::'+p['id'],pdb_id=p['id'].upper(),subset='test_db55',max_contact_residues=p.get('contact_residues',0))
            if not p['valid'] or p.get('interface_status')!='pass' or any(not lookup[p[k]]['valid'] or lookup[p[k]]['missing_residues'] for k in ['receptor','ligand']):
                exclusions.append(exclusion(source,['db55_bound_quality']));continue
            q=dict(p,receptor_row=lookup[p['receptor']],ligand_row=lookup[p['ligand']]);bound.append((source,q))
        if len(bound)!=248:raise ValueError(f'Expected 248 eligible DB5.5 pairs, found {len(bound)}')
        print('Building 248 DB5.5 bound graphs',flush=True)
        for row,p in bound:
            record,error=save_graph(row,'test_db55',output,pair=p)
            if record:manifest.append(record)
            if error:failures.append(error)
        dbids={r['pdb_id'] for r in manifest}
        pool=[]
        for r in eligible:
            if r['pdb_id'].upper() in dbids:exclusions.append(exclusion(r,['pdb_overlap_db55']))
            else:pool.append(r)
        long=[r for r in pool if r['subset']=='snac_db' and len(cdr(r))>=16]
        clusters=cluster_long(long);order=list(range(len(clusters)));random.Random(partition_seed).shuffle(order)
        summary.update(long_eligible=len(long),unique_long_cdr=len({cdr(r) for r in long}),clusters=len(clusters))
        print(f'Hard pool: {len(long)} structures / {len(clusters)} CDR40 clusters',flush=True)
        chosen=[];used_ids=set()
        for index in order:
            cl=clusters[index]
            for row in cl['candidates']:
                record,error=save_graph(row,'test_snac_hard',output,cluster_id=cl['cluster_id'])
                if error:failures.append(error)
                if record:
                    manifest.append(record);chosen.append(row);used_ids.add(row['id']);break
            if not args.no_cap and len(chosen)>=args.target_hard:break
        if args.no_cap:
            if not chosen:raise ValueError('No valid independent challenge graphs found in the qualifying pool (--no-cap)')
        elif len(chosen)!=args.target_hard:raise ValueError(f'Only {len(chosen)} valid independent challenge graphs; requested {args.target_hard}')
        hardseqs=[cdr(r) for r in chosen];hardids={r['pdb_id'].upper() for r in chosen}
        train=[];known_train_cdr={}
        for r in pool:
            if r['id'] in used_ids:continue
            reasons=[]
            if r['pdb_id'].upper() in hardids:reasons.append('pdb_overlap_snac_hard')
            seqs={cdr(r)} if cdr(r) else set()
            if r['subset']=='train_rcsb':seqs.update(x['cdr3'] for x in audit.PDB_ANNOTATIONS.get(r['pdb_id'].upper(),[]) if x['kind']=='VHH' and x['cdr3'])
            if any(seqsim(s,t)>=CLUSTER_IDENTITY_THRESHOLD for s in seqs for t in hardseqs):reasons.append('cdr3_overlap_snac_hard_40pct')
            if reasons:exclusions.append(exclusion(r,reasons))
            else:train.append(r);known_train_cdr[r['id']]=sorted(seqs)
        print(f'Building {len(train)} train graphs; hard test {len(chosen)}',flush=True)
        def worker(row):return save_graph(row,'train',output)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for i,(record,error) in enumerate(executor.map(worker,train),1):
                if record:manifest.append(record)
                if error:failures.append(error)
                if i%200==0:print(f'Train {i}/{len(train)}, elapsed {time.time()-start:.0f}s',flush=True)
        train_records=[r for r in manifest if r['split']=='train']
        assert not ({r['pdb_id'] for r in train_records}&(dbids|hardids))
        assert not (dbids&hardids)
        maxhard=max(seqsim(s,t) for s,t in itertools.combinations(hardseqs,2))
        assert maxhard<CLUSTER_IDENTITY_THRESHOLD
        maxtrain=max((seqsim(s,t) for r in train_records for s in known_train_cdr[r['source_id']] for t in hardseqs),default=0)
        assert maxtrain<CLUSTER_IDENTITY_THRESHOLD
        assert len(list((output/'graphs').rglob('*.pt')))==len(manifest)
        assert sum(r['split']=='test_db55' for r in manifest)==248
        for split in ['train','test_db55','test_snac_hard']:
            sample_rows=[r for r in manifest if r['split']==split][:2]
            sample=[torch.load(output/r['path'],weights_only=False,map_location='cpu') for r in sample_rows]
            batched=Batch.from_data_list(sample);assert batched.num_nodes==sum(g.num_nodes for g in sample)
        summary.update(validation=dict(all_graphs_read_back=True,db55_count_248=True,pdb_split_overlap=0,hard_max_pair_identity=maxhard,known_train_hard_max_identity=maxtrain,pyg_batch=True),known_train_cdr=known_train_cdr)
        cluster_output=[dict(cluster_id=cl['cluster_id'],representative=cl['representative'],sequences=cl['sequences'],source_ids=[r['id'] for r in cl['members']],selected=any(r['cluster_id']==cl['cluster_id'] for r in manifest)) for cl in clusters]
        (output/'cdr3_clusters.json').write_text(json.dumps(cluster_output,ensure_ascii=False,indent=2),encoding='utf-8')
        complete=True
    except Exception as exc:
        failures.append(dict(split='pipeline',source_id='',pdb_id='',reason=type(exc).__name__,detail=str(exc)));traceback.print_exc()
    finally:
        summary.update(elapsed_seconds=previous_elapsed+time.time()-start,complete=complete,graphs=len(manifest),sampled_peak_rss_bytes=PEAK_RSS)
        fields=['split','path','source_id','pdb_id','subset_source','nodes','directed_edges','undirected_edges','density','bytes','cdr3_seq','cdr3_len','num_interface_residues','cluster_id','sha256','identity_resolution_notes']
        write_csv(output/'graph_manifest.csv',manifest,fields)
        (output/'graph_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        write_csv(output/'excluded_samples.csv',exclusions,['source_id','pdb_id','subset_source','reasons'])
        write_csv(output/'processing_failures.csv',failures,['split','source_id','pdb_id','reason','detail'])
        (output/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        delivery_report(output,manifest,exclusions,failures,summary,complete)
        print(f'Delivery report: {output / "graph_dataset_delivery_report.md"}; complete={complete}',flush=True)
    return 0 if complete else 1

if __name__=='__main__':
    raise SystemExit(main())
