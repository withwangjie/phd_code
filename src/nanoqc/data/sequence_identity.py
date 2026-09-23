"""Single Needleman-Wunsch identity kernel shared by every homology gate.

All callers previously carried their own copy of the same alignment:
parasail ``nw_stats_striped_32`` with BLOSUM62, gap open 10 / extend 1,
identity = matches / alignment length (gaps included), evaluated in both
argument orders and symmetrised by ``max``. That kernel now lives here once.

The length-coverage gates, ``a == b`` shortcuts, caching and return types stay
in each caller's wrapper, because they differ on purpose between call sites
(for example ``build_final_pyg_dataset.similarity`` gates on the CDR-H3
threshold, ``run_real_complex_pilot.identity_detail`` returns a record).
"""
from __future__ import annotations

import parasail

GAP_OPEN = 10
GAP_EXTEND = 1


def nw_identity(a: str, b: str, *, saturation_message: str = "alignment score saturation") -> float:
    """Symmetric global identity over alignment length; raise on score saturation."""
    values = []
    for left, right in ((a, b), (b, a)):
        result = parasail.nw_stats_striped_32(left, right, GAP_OPEN, GAP_EXTEND, parasail.blosum62)
        if result.saturated:
            raise ValueError(saturation_message)
        values.append(result.matches / result.length)
    return max(values)
