"""Single Needleman-Wunsch identity kernel shared by every homology gate.

All callers previously carried their own copy of the same alignment:
parasail ``nw_stats_striped_32`` with BLOSUM62, gap open 10 / extend 1,
identity = matches / alignment length (gaps included), evaluated in both
argument orders and symmetrised by ``max``. That kernel now lives here once.

The common length-ratio calculation lives here too. Coverage thresholds,
``a == b`` shortcuts, caching and return types stay in each caller's wrapper,
because they differ on purpose between call sites
(for example ``build_final_pyg_dataset.similarity`` gates on the CDR-H3
threshold, ``run_real_complex_pilot.identity_detail`` returns a record).
"""
from __future__ import annotations

import parasail

GAP_OPEN = 10
GAP_EXTEND = 1


def length_coverage(a: str, b: str) -> float:
    """Shorter/longer sequence length, or zero when either is empty."""
    return min(len(a),len(b))/max(len(a),len(b)) if a and b else 0.0


def nw_identity(a: str, b: str, *, saturation_message: str = "alignment score saturation") -> float:
    """Symmetric global identity over alignment length; raise on score saturation."""
    values = []
    for left, right in ((a, b), (b, a)):
        result = parasail.nw_stats_striped_32(left, right, GAP_OPEN, GAP_EXTEND, parasail.blosum62)
        if result.saturated:
            raise ValueError(saturation_message)
        values.append(result.matches / result.length)
    return max(values)


# Subsets whose group-0 partner is an annotated nanobody chain
# (build_final_pyg_dataset.extract: snac_db anchors chain H, sabdab_vhh anchors
# the unique annotated VHH chain). Every other subset (train_rcsb, extra_*)
# assigns group 0 to the first chain of the strongest contact pair, so its
# VHH/antigen labels are not known to be correct.
ROLE_ANCHORED_SUBSETS = frozenset({"snac_db", "sabdab_vhh"})


def partner_roles_anchored(subset_source: object) -> bool:
    """True when a graph's VHH/antigen partner labels are annotation-anchored."""
    return str(subset_source or "") in ROLE_ANCHORED_SUBSETS


def partner_orientations(left_vhh, left_antigen, left_anchored: bool,
                         right_vhh, right_antigen, right_anchored: bool):
    """Relative partner orientations to test for cross-complex homology.

    Yields ``(left_as_vhh, right_as_vhh, left_as_antigen, right_as_antigen)``.
    The labelled orientation is always tested. For each complex whose roles
    are not anchored, the orientation with *that* complex's partners swapped
    is tested too, so a nanobody stored in the "antigen" slot is still
    compared with the other complex's nanobody under the VHH threshold (and
    its true antigen under the antigen threshold). Swapping both complexes at
    once is the same relative orientation as the labelled one and is not
    repeated; this only ever adds exclusions.
    """
    yield left_vhh, right_vhh, left_antigen, right_antigen
    if not left_anchored:
        yield left_antigen, right_vhh, left_vhh, right_antigen
    if not right_anchored:
        yield left_vhh, right_antigen, left_antigen, right_vhh
