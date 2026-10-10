"""FASPR round trip: unambiguous input, Active-only output in author identities."""
import gemmi
import pytest

from nanoqc.experiments.run_external_structure_baselines import (
    restrict_sidechain_packing, write_faspr_input)


def _residue(name, num, icode, atoms):
    residue = gemmi.Residue(); residue.name = name; residue.seqid = gemmi.SeqId(num, icode)
    for atom_name, element, xyz in atoms:
        atom = gemmi.Atom(); atom.name = atom_name; atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(*xyz); residue.add_atom(atom)
    return residue


def _source(path):
    """Chain H with IMGT 111/112A/112 and a C-terminal OXT; assembly chain A-2."""
    st = gemmi.Structure(); model = gemmi.Model("1")
    heavy = model.add_chain(gemmi.Chain("H")) or model["H"]
    model["H"].add_residue(_residue("SER", 111, " ", [("N","N",(0,0,0)),("CA","C",(1,0,0)),("C","C",(2,0,0)),
        ("O","O",(3,0,0)),("CB","C",(1,1,0)),("OG","O",(1,2,0)),("HG","H",(1,3,0))]))
    model["H"].add_residue(_residue("LYS", 112, "A", [("N","N",(4,0,0)),("CA","C",(5,0,0)),("C","C",(6,0,0)),
        ("O","O",(7,0,0)),("CB","C",(5,1,0)),("CG","C",(5,2,0)),("CD","C",(5,3,0)),("CE","C",(5,4,0)),
        ("NZ","N",(5,5,0))]))
    model["H"].add_residue(_residue("GLY", 112, " ", [("N","N",(8,0,0)),("CA","C",(9,0,0)),("C","C",(10,0,0)),
        ("O","O",(11,0,0)),("OXT","O",(12,0,0))]))
    model.add_chain(gemmi.Chain("A-2"))
    model["A-2"].add_residue(_residue("ALA", 112, " ", [("N","N",(0,9,0)),("CA","C",(1,9,0)),("C","C",(2,9,0)),
        ("O","O",(3,9,0)),("CB","C",(1,10,0))]))
    st.add_model(model); st.setup_entities()
    st.make_mmcif_document().write_file(str(path))


def _fake_faspr(input_pdb, output_pdb, *, drop_ce=False):
    """FASPR's observable behaviour: no OXT, side chains rebuilt (moved)."""
    lines = []
    for line in input_pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name == "OXT" or (drop_ce and name == "CE"):
            continue
        if name not in {"N", "CA", "C", "O"}:
            line = line[:30] + "%8.3f" % (float(line[30:38]) + 0.5) + line[38:]
        lines.append(line)
    output_pdb.write_text("\n".join(lines) + "\nEND\n")


def _atoms(path):
    st = gemmi.read_structure(str(path))
    return {f"{c.name}:{r.seqid}:{a.name}": a.pos.x for c in st[0] for r in c for a in r}


def test_faspr_input_is_unambiguous(tmp_path):
    source, pdb = tmp_path / "in.cif", tmp_path / "in.pdb"
    _source(source)
    order = write_faspr_input(source, pdb)
    assert order == ["H:111:SER", "H:112A:LYS", "H:112:GLY", "A-2:112:ALA"]
    atom_lines = [l for l in pdb.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    assert all(l.startswith("ATOM") for l in atom_lines)
    assert [l[22:27] for l in atom_lines if l[12:16].strip() == "CA"] == ["   1 ", "   2 ", "   3 ", "   4 "]
    assert {l[21] for l in atom_lines} == {"A", "B"}
    assert not any(l[12:16].strip() == "HG" for l in atom_lines)
    assert all(line.strip() for line in pdb.read_text().splitlines())


def test_only_active_sidechains_change_and_author_ids_return(tmp_path):
    source, pdb, packed, out = (tmp_path / n for n in ("in.cif", "in.pdb", "faspr.pdb", "out.cif"))
    _source(source)
    order = write_faspr_input(source, pdb)
    _fake_faspr(pdb, packed)
    restrict_sidechain_packing(source, packed, order, out, ["H:112A"])
    before, after = _atoms(source), _atoms(out)
    assert after["H:112:OXT"] == before["H:112:OXT"]
    assert after["H:112A:CE"] == pytest.approx(before["H:112A:CE"] + 0.5)
    assert after["H:112A:CA"] == before["H:112A:CA"]
    assert after["H:111:OG"] == before["H:111:OG"] and after["A-2:112:CB"] == before["A-2:112:CB"]
    assert "H:111:HG" not in after
    assert set(after) == {k for k in before if not k.endswith(":HG")}


def test_missing_active_atom_stays_missing(tmp_path):
    source, pdb, packed, out = (tmp_path / n for n in ("in.cif", "in.pdb", "faspr.pdb", "out.cif"))
    _source(source)
    order = write_faspr_input(source, pdb)
    _fake_faspr(pdb, packed, drop_ce=True)
    restrict_sidechain_packing(source, packed, order, out, ["H:112A"])
    assert "H:112A:CE" not in _atoms(out) and "H:112:OXT" in _atoms(out)


def test_residue_count_mismatch_is_refused(tmp_path):
    source, pdb, packed, out = (tmp_path / n for n in ("in.cif", "in.pdb", "faspr.pdb", "out.cif"))
    _source(source)
    order = write_faspr_input(source, pdb)
    _fake_faspr(pdb, packed)
    packed.write_text("\n".join(l for l in packed.read_text().splitlines() if l[17:20] != "ALA") + "\n")
    with pytest.raises(ValueError, match="residues"):
        restrict_sidechain_packing(source, packed, order, out, ["H:112A"])
