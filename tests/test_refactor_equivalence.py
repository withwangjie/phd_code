"""Equivalence tests for the stage-1 consolidation of duplicated helpers.

Every helper that was factored into repo_io / sequence_identity /
residue_tables / paired_statistics is compared here against a verbatim copy of
the implementation it replaced (taken from commit 54858e2). Exact equality
(not approximate) is required: these values feed frozen split decisions,
manifests and reported statistics.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import itertools
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

try:  # the references below call parasail exactly as the replaced code did
    import parasail
except ImportError:  # pragma: no cover - the formal environment has parasail
    parasail = None

REPO = Path(__file__).resolve().parents[1]
CDR_H3_IDENTITY_THRESHOLD = 0.50  # value in build_final_pyg_dataset.py @ 54858e2


# ===========================================================================
# Reference implementations (verbatim; only the def/variable name is changed)
# ===========================================================================
# --- verbatim from audit_external_vhh_independence.py @ 54858e2: identity
def ref_external_identity(a: str, b: str, min_length_coverage: float = 0.0) -> tuple[float,float]:
    a,b=str(a or ""),str(b or "")
    if not a or not b:
        return 0.0,0.0
    coverage=min(len(a),len(b))/max(len(a),len(b))
    if coverage < min_length_coverage:
        return 0.0,coverage
    values=[]
    for left,right in ((a,b),(b,a)):
        result=parasail.nw_stats_striped_32(left,right,10,1,parasail.blosum62)
        if result.saturated:
            raise ValueError("parasail alignment saturated")
        values.append(result.matches/result.length)
    return max(values),coverage


# --- verbatim from build_final_pyg_dataset.py @ 54858e2: similarity
@functools.lru_cache(maxsize=300000)
def ref_build_similarity(a,b):
    """Symmetric global identity, exact matches / alignment length incl. gaps."""
    if a==b:return 1.0
    if not a or not b or min(len(a),len(b))/max(len(a),len(b)) < CDR_H3_IDENTITY_THRESHOLD:return 0.0
    r=parasail.nw_stats_striped_32(a,b,10,1,parasail.blosum62)
    s=parasail.nw_stats_striped_32(b,a,10,1,parasail.blosum62)
    if r.saturated or s.saturated:raise ValueError('alignment score saturation')
    return max(r.matches/r.length,s.matches/s.length)


# --- verbatim from build_final_pyg_dataset.py @ 54858e2: _global_identity_cached
@functools.lru_cache(maxsize=500000)
def ref_build_global_identity_cached(a: str, b: str) -> float:
    """Symmetric Needleman-Wunsch identity over alignment length."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    values=[]
    for left,right in ((a,b),(b,a)):
        result=parasail.nw_stats_striped_32(left,right,10,1,parasail.blosum62)
        if result.saturated:
            raise ValueError('alignment score saturation')
        values.append(result.matches/result.length)
    return max(values)


# --- verbatim from build_final_pyg_dataset.py @ 54858e2: global_identity
def ref_build_global_identity(a: str, b: str, *, min_length_coverage: float = 0.0) -> float:
    """Global identity with an explicit pre-alignment length-coverage gate."""
    a,b=str(a or ''),str(b or '')
    if not a or not b:
        return 0.0
    coverage=min(len(a),len(b))/max(len(a),len(b))
    if coverage < min_length_coverage:
        return 0.0
    return ref_build_global_identity_cached(*sorted((a,b)))


# --- verbatim from run_real_complex_pilot.py @ 54858e2: identity_detail
def ref_pilot_identity_detail(a: str, b: str, threshold: float = .4) -> dict:
    """Like `identity`, but always returns the actual identity value and the
    length-ratio coverage gate -- for an auditable record, never just a
    boolean pass/fail. Uses the same alignment (parasail NW + BLOSUM62) and
    the same configurable length-ratio coverage gate as `identity`, just without
    collapsing the result to a threshold comparison."""
    coverage = min(len(a), len(b)) / max(len(a), len(b))
    if coverage < threshold:
        return dict(identity=0., coverage=coverage, length_gated=True)
    values = []
    for x, y in ((a, b), (b, a)):
        result = parasail.nw_stats_striped_32(x, y, 10, 1, parasail.blosum62)
        if result.saturated:
            raise ValueError("Alignment saturation")
        values.append(result.matches / result.length)
    return dict(identity=max(values), coverage=coverage, length_gated=False)


# --- verbatim from train_egnn_pruning.py @ 54858e2: _sequence_identity
def ref_train_sequence_identity(a: str, b: str, *, min_length_coverage: float = 0.0) -> float:
    """Symmetric global identity over alignment length with an explicit length gate."""

    a, b = str(a or ""), str(b or "")
    if not a or not b:
        return 0.0
    coverage = min(len(a), len(b)) / max(len(a), len(b))
    if coverage < min_length_coverage:
        return 0.0
    if a == b:
        return 1.0
    values = []
    for left, right in ((a, b), (b, a)):
        result = parasail.nw_stats_striped_32(left, right, 10, 1, parasail.blosum62)
        if result.saturated:
            raise ValueError("alignment score saturation while building the split")
        values.append(result.matches / result.length)
    return max(values)


# --- verbatim from batch_benchmark_hard_set.py @ 54858e2: _paired_effect
def ref_paired_effect(values: Sequence[float], seed: int = 20260917,
                   resamples: int = 10000) -> dict:
    """Mean paired cluster difference, percentile CI, two-sided sign-flip test.

    Sign exchangeability/symmetry under the null and independent clusters are
    assumptions, not guaranteed by observational benchmark data.
    """
    d = np.asarray(values, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or resamples < 100:
        raise ValueError("Finite nonempty differences and >=100 resamples required")
    result = dict(n_clusters=len(d), mean_difference=float(d.mean()),
                  ci_low=None, ci_high=None, p_value=None)
    if len(d) < 2:
        return result
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, resamples, 256):
        size = min(256, resamples-start)
        boot.extend(d[rng.integers(0,len(d),(size,len(d)))].mean(axis=1))
    result.update(ci_low=float(np.quantile(boot,.025)), ci_high=float(np.quantile(boot,.975)))
    observed = abs(float(d.mean()))
    tolerance = 1e-12*max(1., observed)
    if len(d) <= 16:
        extreme = sum(abs(np.dot(signs,d)/len(d)) >= observed-tolerance
                      for signs in itertools.product((-1,1),repeat=len(d)))
        pvalue = extreme/(2**len(d))
    else:
        extreme = 0
        for start in range(0,resamples,256):
            size = min(256,resamples-start)
            signs = rng.choice([-1.,1.],size=(size,len(d)))
            extreme += int(np.count_nonzero(abs(signs@d/len(d)) >= observed-tolerance))
        pvalue = (extreme+1)/(resamples+1)
    result["p_value"] = float(pvalue)
    return result


# --- verbatim from batch_benchmark_hard_set.py @ 54858e2: _holm_adjust
def ref_holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error adjustment, original ordering retained."""
    p = np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid p values")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.
    for rank,index in enumerate(order):
        running = max(running,(len(p)-rank)*p[index])
        adjusted[index] = min(1.,running)
    return adjusted.tolist()


# --- verbatim from analyze_quantum_scaling.py @ 54858e2: _cluster_effect
def ref_cluster_effect(values:list[float],seed:int,resamples:int)->dict:
    arr=np.asarray(values,float)
    if not len(arr) or not np.isfinite(arr).all():
        return dict(n_clusters=0,mean_slope=None,ci_low=None,ci_high=None,p_value=None)
    result=dict(n_clusters=len(arr),mean_slope=float(arr.mean()),ci_low=None,ci_high=None,p_value=None)
    if len(arr)<2:
        return result
    rng=np.random.default_rng(seed)
    boots=[]
    for start in range(0,resamples,256):
        size=min(256,resamples-start)
        boots.extend(arr[rng.integers(0,len(arr),(size,len(arr)))].mean(axis=1))
    result["ci_low"]=float(np.quantile(boots,.025))
    result["ci_high"]=float(np.quantile(boots,.975))
    observed=abs(float(arr.mean()))
    tol=1e-12*max(1.0,observed)
    if len(arr)<=16:
        extreme=sum(
            abs(float(np.dot(signs,arr)/len(arr)))>=observed-tol
            for signs in itertools.product((-1,1),repeat=len(arr))
        )
        p=extreme/(2**len(arr))
    else:
        extreme=0
        for start in range(0,resamples,256):
            size=min(256,resamples-start)
            signs=rng.choice([-1.,1.],size=(size,len(arr)))
            extreme += int(np.count_nonzero(np.abs(signs@arr/len(arr))>=observed-tol))
        p=(extreme+1)/(resamples+1)
    result["p_value"]=float(p)
    return result


# --- verbatim from analyze_structure_recovery.py @ 54858e2: holm_adjust_named
def ref_holm_adjust_named(pvalues: dict[str,float|None]) -> dict[str,float|None]:
    """Holm step-down adjustment for the pre-registered confirmatory family."""
    valid=sorted(
        ((name,float(value)) for name,value in pvalues.items()
         if value is not None and math.isfinite(float(value))),
        key=lambda item:item[1],
    )
    adjusted={name:None for name in pvalues}
    running=0.0
    m=len(valid)
    for rank,(name,value) in enumerate(valid):
        candidate=min(1.0,(m-rank)*value)
        running=max(running,candidate)
        adjusted[name]=running
    return adjusted


# --- verbatim from subgraph_to_qubo.py @ 54858e2: _SIDECHAIN_NAMES
REF_QUBO_SIDECHAIN_NAMES = {
    "GLY": "", "ALA": "CB", "SER": "CB OG", "CYS": "CB SG",
    "THR": "CB OG1 CG2", "VAL": "CB CG1 CG2", "ILE": "CB CG1 CG2 CD1",
    "LEU": "CB CG CD1 CD2", "ASP": "CB CG OD1 OD2", "ASN": "CB CG OD1 ND2",
    "GLU": "CB CG CD OE1 OE2", "GLN": "CB CG CD OE1 NE2",
    "LYS": "CB CG CD CE NZ", "ARG": "CB CG CD NE CZ NH1 NH2",
    "MET": "CB CG SD CE", "PRO": "CB CG CD", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "PHE": "CB CG CD1 CD2 CE1 CE2 CZ", "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH",
    "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
}


# --- verbatim from subgraph_to_qubo.py @ 54858e2: _SYMMETRIC_SWAPS
# (amended: prochiral VAL CG1/CG2 and LEU CD1/CD2 are no longer treated as symmetric)
REF_QUBO_SYMMETRIC_SWAPS = {
    "ASP": [("OD1", "OD2")], "GLU": [("OE1", "OE2")],
    "ARG": [("NH1", "NH2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}


# --- verbatim from audit_all_datasets.py @ 54858e2: SIDECHAIN_HEAVY
REF_AUDIT_SIDECHAIN_HEAVY = {
    'ALA': {'CB'}, 'CYS': {'CB','SG'}, 'ASP': {'CB','CG','OD1','OD2'},
    'GLU': {'CB','CG','CD','OE1','OE2'}, 'PHE': {'CB','CG','CD1','CD2','CE1','CE2','CZ'},
    'GLY': set(), 'HIS': {'CB','CG','ND1','CD2','CE1','NE2'},
    'ILE': {'CB','CG1','CG2','CD1'}, 'LYS': {'CB','CG','CD','CE','NZ'},
    'LEU': {'CB','CG','CD1','CD2'}, 'MET': {'CB','CG','SD','CE'},
    'ASN': {'CB','CG','OD1','ND2'}, 'PRO': {'CB','CG','CD'},
    'GLN': {'CB','CG','CD','OE1','NE2'}, 'ARG': {'CB','CG','CD','NE','CZ','NH1','NH2'},
    'SER': {'CB','OG'}, 'THR': {'CB','OG1','CG2'}, 'VAL': {'CB','CG1','CG2'},
    'TRP': {'CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'},
    'TYR': {'CB','CG','CD1','CD2','CE1','CE2','CZ','OH'},
}


# --- verbatim from audit_all_datasets.py @ 54858e2: BACKBONE
REF_AUDIT_BACKBONE = {'N', 'CA', 'C', 'O'}


# --- verbatim from evaluate_complex_metrics.py @ 54858e2: _SIDECHAIN_HEAVY_ATOMS
REF_EVAL_SIDECHAIN_HEAVY_ATOMS: Dict[str, Tuple[str, ...]] = {
    "GLY": (), "ALA": ("CB",), "SER": ("CB", "OG"), "CYS": ("CB", "SG"),
    "THR": ("CB", "OG1", "CG2"), "VAL": ("CB", "CG1", "CG2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"), "LEU": ("CB", "CG", "CD1", "CD2"),
    "ASP": ("CB", "CG", "OD1", "OD2"), "ASN": ("CB", "CG", "OD1", "ND2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"), "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "MET": ("CB", "CG", "SD", "CE"), "PRO": ("CB", "CG", "CD"),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}


# --- verbatim from evaluate_complex_metrics.py @ 54858e2: _SYMMETRIC_SWAPS
# (amended: prochiral VAL CG1/CG2 and LEU CD1/CD2 are no longer treated as symmetric)
REF_EVAL_SYMMETRIC_SWAPS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ASP": (("OD1", "OD2"),), "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
}


# --- verbatim from evaluate_complex_metrics.py @ 54858e2: BACKBONE_ATOMS
REF_EVAL_BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")




# ===========================================================================
# repo_io
# ===========================================================================

def _ref_chunked_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_sha256_file_matches_streamed_reference(tmp_path):
    from nanoqc.common.repo_io import sha256_file
    rng = random.Random(7)
    for size in (0, 1, (1 << 20) - 1, 1 << 20, (1 << 20) + 17, 3 * (1 << 20) + 5):
        path = tmp_path / f"blob_{size}.bin"
        path.write_bytes(bytes(rng.getrandbits(8) for _ in range(size)) if size < 5000
                         else rng.randbytes(size))
        assert sha256_file(path) == _ref_chunked_sha256(path)
        assert sha256_file(str(path)) == _ref_chunked_sha256(path)


def test_sha256_file_without_python_311_file_digest(tmp_path, monkeypatch):
    from nanoqc.common.repo_io import sha256_file

    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    path = tmp_path / "python310.bin"
    path.write_bytes(b"Python 3.10 compatibility\x00")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_json_fsync_format_unchanged(tmp_path):
    import json
    from nanoqc.common.repo_io import atomic_write_json_fsync
    value = {"b": [1, 2.5, None], "a": {"x": "y"}}
    path = tmp_path / "record.json"
    atomic_write_json_fsync(path, value)
    assert path.read_text(encoding="utf-8") == json.dumps(value, indent=2, allow_nan=False)
    assert not path.with_suffix(".json.tmp").exists()
    with pytest.raises(ValueError):
        atomic_write_json_fsync(tmp_path / "nan.json", {"x": float("nan")})


# ===========================================================================
# sequence_identity (requires parasail, as in the formal environment)
# ===========================================================================

def _random_pairs(n: int = 300):
    rng = random.Random(20260917)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    pairs = [("", ""), ("", "ACD"), ("ACDE", "ACDE"), ("QVQLVESGGG", "QVQLVESGGG"[:3])]
    for _ in range(n):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 140)))
        if rng.random() < 0.5:
            b = list(a)
            for _ in range(rng.randint(0, max(1, len(b) // 3))):
                op = rng.random()
                pos = rng.randrange(len(b)) if b else 0
                if op < 0.5 and b:
                    b[pos] = rng.choice(alphabet)
                elif op < 0.75:
                    b.insert(pos, rng.choice(alphabet))
                elif b:
                    del b[pos]
            b = "".join(b) or "A"
        else:
            b = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 140)))
        pairs.append((a, b))
    return pairs


def test_identity_wrappers_match_references():
    if parasail is None:
        pytest.skip("parasail is not installed")
    import nanoqc.data.audit_external_vhh_independence as ext
    import nanoqc.data.build_final_pyg_dataset as build
    import nanoqc.experiments.run_real_complex_pilot as pilot
    import nanoqc.model.train_egnn_pruning as train
    build.CDR_H3_IDENTITY_THRESHOLD = CDR_H3_IDENTITY_THRESHOLD
    build.similarity.cache_clear()
    build._global_identity_cached.cache_clear()
    for a, b in _random_pairs():
        for cov in (0.0, 0.3, 0.7, 0.95):
            assert ext.identity(a, b, cov) == ref_external_identity(a, b, cov)
            assert build.global_identity(a, b, min_length_coverage=cov) == \
                ref_build_global_identity(a, b, min_length_coverage=cov)
            assert train._sequence_identity(a, b, min_length_coverage=cov) == \
                ref_train_sequence_identity(a, b, min_length_coverage=cov)
        assert build.similarity(a, b) == ref_build_similarity(a, b)
        if a and b:
            for threshold in (0.0, 0.4, 0.8):
                assert pilot.identity_detail(a, b, threshold) == ref_pilot_identity_detail(a, b, threshold)


# ===========================================================================
# residue_tables
# ===========================================================================

def test_residue_tables_reproduce_every_former_copy():
    import nanoqc.data.audit_all_datasets as audit
    import nanoqc.structure.evaluate_complex_metrics as ev
    import nanoqc.qubo.subgraph_to_qubo as sq
    assert sq._SIDECHAIN_NAMES == REF_QUBO_SIDECHAIN_NAMES
    assert all(type(v) is str for v in sq._SIDECHAIN_NAMES.values())
    assert {k: v.split() for k, v in sq._SIDECHAIN_NAMES.items()} == \
        {k: v.split() for k, v in REF_QUBO_SIDECHAIN_NAMES.items()}
    assert sq._SYMMETRIC_SWAPS == REF_QUBO_SYMMETRIC_SWAPS
    assert all(type(v) is list for v in sq._SYMMETRIC_SWAPS.values())
    assert audit.SIDECHAIN_HEAVY == REF_AUDIT_SIDECHAIN_HEAVY
    assert audit.BACKBONE == REF_AUDIT_BACKBONE
    assert ev._SIDECHAIN_HEAVY_ATOMS == REF_EVAL_SIDECHAIN_HEAVY_ATOMS
    assert ev._SYMMETRIC_SWAPS == REF_EVAL_SYMMETRIC_SWAPS
    assert ev.BACKBONE_ATOMS == REF_EVAL_BACKBONE_ATOMS


# ===========================================================================
# paired_statistics
# ===========================================================================

def _stat_inputs():
    rng = np.random.default_rng(11)
    cases = [[0.3], [0.1, -0.2], [1.0, 1.0, 1.0], [0.0] * 5]
    for n in (2, 3, 7, 16, 17, 25, 60):
        cases.append(rng.normal(0.1, 1.0, n).tolist())
        cases.append(np.round(rng.normal(0.0, 1.0, n), 1).tolist())  # ties
    return cases


def test_paired_effect_and_cluster_effect_match_references():
    import nanoqc.inference.analyze_quantum_scaling as scaling
    import nanoqc.experiments.batch_benchmark_hard_set as bbh
    from nanoqc.inference.paired_statistics import paired_effect
    assert bbh._paired_effect is paired_effect
    for values in _stat_inputs():
        for seed, resamples in ((20260917, 10000), (5, 300)):
            assert paired_effect(values, seed, resamples) == ref_paired_effect(values, seed, resamples)
            assert scaling._cluster_effect(values, seed, resamples) == \
                ref_cluster_effect(values, seed, resamples)
    assert scaling._cluster_effect([], 1, 1000) == ref_cluster_effect([], 1, 1000)
    assert scaling._cluster_effect([1.0, float("nan")], 1, 1000) == \
        ref_cluster_effect([1.0, float("nan")], 1, 1000)
    for bad in ([], [float("inf"), 1.0]):
        with pytest.raises(ValueError):
            paired_effect(bad, 1, 1000)
    with pytest.raises(ValueError):
        paired_effect([1.0, 2.0], 1, 50)


def test_holm_variants_match_references():
    import nanoqc.inference.analyze_structure_recovery as structure
    from nanoqc.inference.paired_statistics import holm_adjust
    rng = np.random.default_rng(3)
    families = [[], [0.5], [0.01, 0.04, 0.03, 0.2], [0.02, 0.02, 0.02], [1.0, 0.0, 0.5, 0.5]]
    families += [np.round(rng.uniform(0, 1, n), 2).tolist() for n in (2, 5, 9, 30)]
    for family in families:
        assert holm_adjust(family) == ref_holm_adjust(family)
        named = {f"h{i}": p for i, p in enumerate(family)}
        named["missing"] = None
        named["nan"] = float("nan")
        assert structure.holm_adjust_named(named) == ref_holm_adjust_named(named)
    for bad in ([0.1, float("nan")], [1.2], [-0.1]):
        with pytest.raises(ValueError):
            holm_adjust(bad)


# ===========================================================================
# Run-level code fingerprint completeness (run_full_experiment)
# ===========================================================================

def _local_import_closure(start):
    from nanoqc.common.repo_io import MODULE_LAYOUT, repo_path
    by_dotted = {path[len("src/"):-len(".py")].replace("/", "."): name
                 for name, path in MODULE_LAYOUT.items()}
    seen, todo = set(), list(start)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse(repo_path(name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            todo.extend(by_dotted[m] for m in modules if m in by_dotted)
    return seen


def test_module_layout_matches_the_source_tree():
    from nanoqc.common.repo_io import MODULE_LAYOUT
    on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / "src" / "nanoqc").rglob("*.py")
               if p.name != "__init__.py"}
    assert set(MODULE_LAYOUT.values()) == on_disk
    assert all(Path(path).name == name for name, path in MODULE_LAYOUT.items())


def test_orchestrated_scripts_cover_every_executed_local_module():
    import nanoqc.pipeline.run_full_experiment as full
    from nanoqc.common.repo_io import repo_path
    closure = _local_import_closure(full.ORCHESTRATED_SCRIPTS)
    assert closure <= set(full.ORCHESTRATED_SCRIPTS), sorted(closure - set(full.ORCHESTRATED_SCRIPTS))
    for name in full.ORCHESTRATED_SCRIPTS:
        assert repo_path(name).is_file(), name


def test_benchmark_code_fingerprint_covers_quantum_instance_modules():
    from nanoqc.experiments.batch_benchmark_hard_set import SHARED_HELPER_MODULES
    assert {"instance.py", "resource_estimation.py"} <= set(SHARED_HELPER_MODULES)


def test_orchestrator_never_passes_the_bare_master_seed_to_statistics():
    source = (REPO / "src/nanoqc/pipeline/run_full_experiment.py").read_text(encoding="utf-8")
    assert '"--seed",str(self.config["master_seed"])' not in source
    assert '"--seed", str(self.config["master_seed"])' not in source


def test_build_run_manifest_fails_closed_on_missing_script(tmp_path):
    import nanoqc.pipeline.run_full_experiment as full
    from nanoqc.common.repo_io import repo_path

    def stub(name):
        path = repo_path(name, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")

    for name in full.ORCHESTRATED_SCRIPTS[1:]:
        stub(name)
    with pytest.raises(FileNotFoundError):
        full.build_run_manifest({"master_seed": 1}, tmp_path)
    stub(full.ORCHESTRATED_SCRIPTS[0])
    manifest = full.build_run_manifest({"master_seed": 1}, tmp_path)
    assert set(manifest["code_sha256"]) == set(full.ORCHESTRATED_SCRIPTS)
