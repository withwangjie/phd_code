"""The formal provider consumes Rosetta sample statistics without text fallback."""

import sys
from types import SimpleNamespace

import pytest

from nanoqc.qubo.subgraph_to_qubo import (
    _load_rotamer_bins, _dunbrack_templates_for_site, rotamer_source_metadata,
)


def test_pyrosetta_dun10_samples_drive_candidate_statistics(monkeypatch):
    calls = []

    class Sample:
        def nchi(self):
            return 2

        def probability(self):
            return 0.4

        def chi_mean(self):
            return [None, -70.0, 180.0]

        def chi_sd(self):
            return [None, 5.0, 8.0]

    class Library:
        def get_all_rotamer_samples(self, backbone):
            calls.append((backbone[1], backbone[2]))
            return [Sample()]

    rosetta = SimpleNamespace(
        basic=SimpleNamespace(
            was_init_called=lambda: False,
            options=SimpleNamespace(get_boolean_option=lambda name: name == "dun10"),
        ),
        core=SimpleNamespace(
            chemical=SimpleNamespace(aa_from_oneletter_code=lambda aa: aa),
            pack=SimpleNamespace(dunbrack=SimpleNamespace(
                RotamerLibrary=SimpleNamespace(get_instance=lambda: SimpleNamespace(
                    get_library_by_aa=lambda aa: Library())))),
        ),
        utility=SimpleNamespace(fixedsizearray1_double_5_t=lambda: [0.0] * 6),
    )
    fake = SimpleNamespace(rosetta=rosetta,
                           init=lambda options: calls.append(options),
                           version=lambda: "PyRosetta-4 2026.29-test")
    monkeypatch.setitem(sys.modules, "pyrosetta", fake)

    bins = _load_rotamer_bins("pyrosetta_dun10", None, {("LYS", -60, -40)})
    assert calls == ["-mute all -dun10", (-60.0, -40.0)]
    row = bins[("LYS", -60, -40)][0]
    assert row.source == "pyrosetta_dun10"
    assert row.chi_degrees == (-70.0, 180.0)
    assert row.chi_sigmas == (5.0, 8.0)
    assert row.prior_probability == 1.0
    expanded = _dunbrack_templates_for_site(
        bins, "K", -60, -40, probability_floor=1e-4, sigma_offsets=(-1, 0, 1))
    assert {row.chi1_degrees for row in expanded} == {-75.0, -70.0, -65.0}
    assert {row.source for row in expanded} == {"pyrosetta_dun10"}
    assert rotamer_source_metadata("pyrosetta_dun10", None)["pyrosetta_version"] == "PyRosetta-4 2026.29-test"


def test_pyrosetta_mode_fails_closed_without_installation(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrosetta", None)
    with pytest.raises(RuntimeError, match="requires an installed PyRosetta"):
        _load_rotamer_bins("pyrosetta_dun10", None, {("LYS", -60, -40)})
