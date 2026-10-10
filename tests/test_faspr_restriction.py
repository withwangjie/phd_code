"""Active-only FASPR restriction restores atoms FASPR does not own."""
import gemmi

from nanoqc.experiments.run_external_structure_baselines import restrict_sidechain_packing


def _pdb(path, atoms):
    lines=[]
    for serial,(res,seq,name,xyz,el) in enumerate(atoms,1):
        lines.append("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00 20.00          %2s"
                     % (serial,name if len(name)==4 else " "+name,res,seq,*xyz,el))
    path.write_text("\n".join(lines+["END"])+"\n")


def _atoms(seq_extra_oxt, ser_og=True, lys_ce=True):
    atoms=[("SER",1,n,(i,0,0),n[0]) for i,n in enumerate(["N","CA","C","O","CB"])]
    if ser_og:
        atoms.append(("SER",1,"OG",(5,0,0),"O"))
    atoms+=[("LYS",2,n,(i,2,0),n[0]) for i,n in enumerate(["N","CA","C","O","CB","CG","CD"])]
    if lys_ce:
        atoms.append(("LYS",2,"CE",(7,2,0),"C"))
    atoms.append(("LYS",2,"NZ",(8,2,0),"N"))
    if seq_extra_oxt:
        atoms.append(("LYS",2,"OXT",(9,2,0),"O"))
    return atoms


def _names(path):
    st=gemmi.read_structure(str(path))
    return {f"{r.seqid.num}:{a.name}" for r in st[0]["A"] for a in r}


def test_oxt_and_nonactive_atoms_restored(tmp_path):
    source,packed,out=tmp_path/"in.pdb",tmp_path/"faspr.pdb",tmp_path/"out.cif"
    _pdb(source,_atoms(True))
    _pdb(packed,_atoms(False,ser_og=False))  # FASPR drops OXT and (here) SER OG
    restrict_sidechain_packing(source,packed,out,["A:2"])
    names=_names(out)
    assert "2:OXT" in names and "1:OG" in names


def test_missing_active_sidechain_atom_is_not_filled(tmp_path):
    source,packed,out=tmp_path/"in.pdb",tmp_path/"faspr.pdb",tmp_path/"out.cif"
    _pdb(source,_atoms(True))
    _pdb(packed,_atoms(False,lys_ce=False))
    restrict_sidechain_packing(source,packed,out,["A:2"])
    names=_names(out)
    assert "2:OXT" in names and "2:CE" not in names
