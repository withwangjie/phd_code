"""Strict author-ID matched docking metrics; explicit custom/standard conventions."""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree


def docking_quality(ref: dict, pred: dict, aligned: dict, active: list,
                    partners: list, rigid_fit) -> dict:
    """Return requested receptor-aligned heavy-atom DockQ variant and BB-fit variant.

    The requested heavy-atom iRMSD is NOT the published DockQ backbone-interface
    superposition. Both are named explicitly. Clash count is not MolProbity.
    """
    bb=('N','CA','C','O')
    ligand_chains={r.rsplit(':',1)[0] for r in active}
    ligand=[r for r in ref if r.rsplit(':',1)[0] in ligand_chains]
    selected=ligand+list(partners)
    for r in selected:
        if r not in pred or ref[r]['name']!=pred[r]['name']:
            raise ValueError(f'Docking residue mismatch: {r}')
        if set(ref[r]['atoms']) != set(pred[r]['atoms']):
            raise ValueError(f'Docking heavy atom mismatch: {r}')

    def contacts(structure, cutoff):
        a=[(r,x) for r in ligand for x in structure[r]['atoms'].values()]
        b=[(r,x) for r in partners for x in structure[r]['atoms'].values()]
        tree=cKDTree([x for _,x in b]); pairs=set()
        for r,x in a:
            for j in tree.query_ball_point(x,cutoff):
                if np.linalg.norm(x-b[j][1]) < cutoff:
                    pairs.add((r,b[j][0]))
        return pairs
    native=contacts(ref,5.); predicted=contacts(pred,5.)
    interface=sorted({r for pair in contacts(ref,10.) for r in pair})
    fnat=len(native & predicted)/len(native) if native else None
    def arrays(ids,names=None):
        keys=[(r,n) for r in ids for n in (names or sorted(ref[r]['atoms']))]
        return np.array([aligned[r][n] for r,n in keys]),np.array([ref[r]['atoms'][n] for r,n in keys])
    lm,lf=arrays(ligand,bb)
    lrmsd=float(np.sqrt(np.mean(np.sum((lm-lf)**2,axis=1))))
    irmsd=standard=None
    if interface:
        m,f=arrays(interface)
        irmsd=float(np.sqrt(np.mean(np.sum((m-f)**2,axis=1))))
        m,f=arrays(interface,bb); rotation,translation=rigid_fit(m,f)
        standard=float(np.sqrt(np.mean(np.sum((m@rotation+translation-f)**2,axis=1))))
    def score(i):
        return (fnat+1/(1+(i/1.5)**2)+1/(1+(lrmsd/8.5)**2))/3 if fnat is not None and i is not None else None
    dockq=score(irmsd)
    category=None if dockq is None else ('Incorrect' if dockq<.23 else 'Acceptable' if dockq<.49 else 'Medium' if dockq<.8 else 'High')
    # Peptide adjacency follows reference author order and a reference C-N bond.
    exclusions=set()
    chains={r.rsplit(':',1)[0] for r in ref}
    for chain in chains:
        ids=[r for r in ref if r.rsplit(':',1)[0]==chain]
        for a,b in zip(ids,ids[1:]):
            if 'C' in ref[a]['atoms'] and 'N' in ref[b]['atoms'] and np.linalg.norm(ref[a]['atoms']['C']-ref[b]['atoms']['N'])<1.9:
                exclusions.add(frozenset(((a,'C'),(b,'N'))))
    sulfurs=[r for r in ref if ref[r]['name']=='CYS' and 'SG' in ref[r]['atoms']]
    for i,a in enumerate(sulfurs):
        for b in sulfurs[i+1:]:
            if np.linalg.norm(ref[a]['atoms']['SG']-ref[b]['atoms']['SG'])<2.3:
                exclusions.add(frozenset(((a,'SG'),(b,'SG'))))
    atoms=[(r,n,x) for r in pred for n,x in pred[r]['atoms'].items()]
    xyz=np.array([a[2] for a in atoms]); severe=[]
    for i,j in sorted(cKDTree(xyz).query_pairs(1.5)):
        a,b=atoms[i],atoms[j]
        if a[0]==b[0] or frozenset((a[:2],b[:2])) in exclusions:
            continue
        if np.linalg.norm(a[2]-b[2])<1.5:
            severe.append([a[0],a[1],b[0],b[1]])
    return dict(dockq_score=dockq,fnat=fnat,irmsd=irmsd,lrmsd=lrmsd,dockq_category=category,
        num_severe_clashes=len(severe),has_severe_clash=bool(severe),severe_clash_pairs=severe,
        dockq_definition='requested receptor-aligned interface-heavy-atom variant; not official DockQ',
        dockq_backbone_score=score(standard),irmsd_interface_backbone_fit=standard,
        docking_native_contacts=len(native),docking_predicted_contacts=len(predicted),
        docking_interface_residues=len(interface),clash_definition='heavy atom distance <1.5 A; same-residue, reference peptide C-N and disulfide S-S excluded; not MolProbity')
