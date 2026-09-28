"""Atom/residue extraction from audited complexes and antigen partner-chain selection.

Split out of build_final_pyg_dataset.py, which re-exports every name here.
Graph assembly (``make_graph``), validation and homology checks stay there
because they read the thresholds that ``main`` rebinds from the command line.
"""
from __future__ import annotations

import collections
import math
import pathlib
import gemmi
import numpy as np
from scipy.spatial import cKDTree
import nanoqc.data.audit_all_datasets as audit
from nanoqc.structure.residue_tables import PEPTIDE_BOND_MAX_C_N_ANGSTROM

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {a:i for i,a in enumerate(AA)}
# SAbDab antigen-chain criterion [METHODS_EVIDENCE R33]: any CA/CB within 7.5 A
# of a CA/CB of the antibody's CDR residues.
ANTIGEN_CHAIN_CONTACT_ANGSTROM = 7.5

def cdr(row):
    seqs=row.get('cdr3_sequences',[])
    return seqs[0] if len(seqs)==1 else ''

def _dihedral_degrees(a, b, c, d):
    """Signed torsion in degrees for four Cartesian points."""

    p0,p1,p2,p3=(np.asarray(x,dtype=np.float64) for x in (a,b,c,d))
    b0=-(p1-p0); b1=p2-p1; b2=p3-p2
    norm=np.linalg.norm(b1)
    if norm <= 1e-12:
        return math.nan
    b1=b1/norm
    v=b0-np.dot(b0,b1)*b1
    w=b2-np.dot(b2,b1)*b1
    if np.linalg.norm(v)<=1e-12 or np.linalg.norm(w)<=1e-12:
        return math.nan
    x=np.dot(v,w); y=np.dot(np.cross(b1,v),w)
    return float(np.degrees(np.arctan2(y,x)))


def _peptide_bonded(previous, following):
    """True when C(previous)-N(following) is a real peptide bond, not a chain break."""
    gap=np.asarray(previous['c'],dtype=np.float64)-np.asarray(following['n'],dtype=np.float64)
    return float(np.linalg.norm(gap))<PEPTIDE_BOND_MAX_C_N_ANGSTROM

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
            cb=atoms['CB'].pos if 'CB' in atoms else None
            nodes.append(dict(
                aa=aa,pos=(ca.x,ca.y,ca.z),residue_id=prefix+chain.name+':'+key,name=res.name,
                cb=(None if cb is None else (cb.x,cb.y,cb.z)),
                n=(atoms['N'].pos.x,atoms['N'].pos.y,atoms['N'].pos.z),
                c=(atoms['C'].pos.x,atoms['C'].pos.y,atoms['C'].pos.z),
                phi=math.nan,psi=math.nan,
            ))
            for atom in atoms.values():
                heavy.append((atom.pos.x,atom.pos.y,atom.pos.z));owners.append(len(nodes)-1)
        for idx,node in enumerate(nodes):
            ca=np.asarray(node['pos'],dtype=np.float64)
            # phi/psi need a real peptide bond to the neighbour; at a chain
            # break (unresolved residues) they stay NaN, which later excludes
            # the residue from Dunbrack site selection instead of looking up a
            # meaningless backbone bin.
            if idx>0 and _peptide_bonded(nodes[idx-1],node):
                node['phi']=_dihedral_degrees(nodes[idx-1]['c'],node['n'],ca,node['c'])
            if idx+1<len(nodes) and _peptide_bonded(node,nodes[idx+1]):
                node['psi']=_dihedral_degrees(node['n'],ca,node['c'],nodes[idx+1]['n'])
        if nodes:chains.append(dict(name=prefix+chain.name,original_name=chain.name,nodes=nodes,xyz=np.array(heavy),owners=np.array(owners,dtype=np.int64)))
    if not chains:raise ValueError('no amino acid nodes')
    if len({c['name'] for c in chains})!=len(chains):raise ValueError('ambiguous duplicate chain IDs')
    return chains

def task_of(row):
    return {k:row[k] for k in ['path','member','subset','id']}


def verify_audited_source(row):
    """Fail closed if raw structure bytes changed after the data audit."""
    expected = str(row.get('source_structure_sha256') or '')
    if not expected:
        raise ValueError(f'audited source SHA-256 missing for {row.get("id", row.get("pdb_id", ""))}')
    actual = audit.task_source_sha256(task_of(row))
    if actual != expected:
        raise ValueError(
            f'raw structure changed since data audit for {row.get("id", row.get("pdb_id", ""))}: '
            f'expected {expected}, observed {actual}'
        )
    return actual


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
    if row['subset'] in ('snac_db','sabdab_vhh'):
        anchor=str(row.get('vhh_chain') or ('H' if row['subset']=='snac_db' else ''))
        if not anchor:raise ValueError('strict VHH anchor missing from audited provenance')
    else:
        # Generic complexes remain supported for standalone/debug utilities,
        # but train_rcsb is audit-only and never reaches formal graph building.
        if not row['pairs']:raise ValueError('missing audited contact pairs')
        strongest=sorted(row['pairs'],key=lambda p:(-(p[2]+p[3]),p[0],p[1]))[0]
        anchor=strongest[0]
    if sum(c['name']==anchor for c in chains)!=1:raise ValueError('anchor chain absent or ambiguous')
    for c in chains:c['group']=0 if c['name']==anchor else 1
    if {c['group'] for c in chains}!={0,1}:raise ValueError('both interaction partners required')
    meta=dict(structure_source=str(dict(st.info).get(audit.STRUCTURE_SOURCE_KEY,'as_deposited_file')),
              antigen_chain_rule='all non-VHH chains of the curated complex file',
              antigen_contact_basis='not_applicable',dropped_chains=[])
    if row['subset'] in audit.ASSEMBLY_SUBSETS:
        chains,meta=select_contacting_partner_chains(chains,row,meta)
    for c in chains:c['complex_meta']=meta
    return chains


def _ca_cb_coordinates(nodes):
    return np.asarray([xyz for node in nodes for xyz in (node['pos'],node.get('cb')) if xyz is not None],dtype=np.float64)


def _paratope_nodes(anchor,row):
    """VHH CDR1-3 residues when the audited source maps them, else whole VHH."""
    nodes=anchor['nodes']
    if row['subset']!='sabdab_vhh':
        return nodes,'whole_anchor_chain'
    sequence=''.join(node['aa'] for node in nodes)
    cdr1=(row.get('cdr1_sequences') or [''])[0]
    cdr2=(row.get('cdr2_sequences') or [''])[0]
    cdr3=cdr(row)
    loops=[cdr1,cdr2,cdr3]
    if all(loops) and all(sequence.count(loop)==1 for loop in loops):
        indices=sorted({i for loop in loops for i in range(sequence.index(loop),sequence.index(loop)+len(loop))})
        return [nodes[i] for i in indices],'vhh_cdr1_cdr2_cdr3'
    return nodes,'whole_vhh_chain_cdr_unmapped'


def select_contacting_partner_chains(chains,row,meta):
    """Keep only partner chains that contact the anchor's paratope (SAbDab rule).

    A partner chain is antigen when any of its CA/CB atoms lies within
    ANTIGEN_CHAIN_CONTACT_ANGSTROM of a CA/CB of the paratope residues. For
    sabdab_vhh, further copies of the VHH itself are never antigen.
    """
    anchor=next(c for c in chains if c['group']==0)
    paratope,basis=_paratope_nodes(anchor,row)
    tree=cKDTree(_ca_cb_coordinates(paratope))
    anchor_sequence=''.join(node['aa'] for node in anchor['nodes'])
    kept=[];dropped=[]
    for chain in chains:
        if chain['group']==0:
            kept.append(chain);continue
        sequence=''.join(node['aa'] for node in chain['nodes'])
        if row['subset']=='sabdab_vhh' and sequence==anchor_sequence:
            dropped.append(dict(chain=chain['name'],reason='copy_of_vhh'));continue
        if row['subset']=='sabdab_vhh':
            author=audit._author_chain(chain['name'])
            other_antibody=set(row.get('other_antibody_chains') or [])
            annotated_antigen=set(row.get('sabdab_antigen_chains') or [])
            if author in other_antibody:
                dropped.append(dict(chain=chain['name'],reason='other_antibody_chain'));continue
            if annotated_antigen and author not in annotated_antigen:
                dropped.append(dict(chain=chain['name'],reason='not_sabdab_annotated_antigen'));continue
        distance=float(tree.query(_ca_cb_coordinates(chain['nodes']),k=1)[0].min())
        if distance<=ANTIGEN_CHAIN_CONTACT_ANGSTROM:
            kept.append(chain)
        else:
            dropped.append(dict(chain=chain['name'],reason=f'no_paratope_contact_within_{ANTIGEN_CHAIN_CONTACT_ANGSTROM:g}A',
                                min_ca_cb_distance=round(distance,3)))
    if not any(c['group']==1 for c in kept):
        raise ValueError(f'no partner chain within {ANTIGEN_CHAIN_CONTACT_ANGSTROM:g} A of the paratope in the biological assembly')
    meta=dict(meta,antigen_chain_rule=(f'biological-assembly chains with any CA/CB within '
                                       f'{ANTIGEN_CHAIN_CONTACT_ANGSTROM:g} A of a paratope CA/CB (SAbDab)'),
              antigen_contact_basis=basis,dropped_chains=dropped)
    return kept,meta
