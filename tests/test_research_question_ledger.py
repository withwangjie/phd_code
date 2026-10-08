"""The research-question ledger applies one pre-declared verdict rule."""
import json
from pathlib import Path

from nanoqc.reporting.report_common import ReportContext
from nanoqc.reporting.report_question_ledger import section_research_question_ledger, verdict


def test_verdict_rule() -> None:
    kw = dict(positive="up", negative="down")
    assert verdict(0.01, 2.0, 0.05, **kw).endswith("up")
    assert verdict(0.01, -2.0, 0.05, **kw).endswith("down")
    assert verdict(0.2, 2.0, 0.05, **kw).startswith("null not rejected")
    assert verdict(None, 2.0, 0.05, **kw).startswith("not testable")
    assert verdict(0.01, 2.0, 0.05, evidence_complete=False, **kw).startswith("not established")


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(tmp_path: Path, *, primary_p: float, slope_p: float, completed=("qc_benchmark", "statistics",
                                                                        "structure_experiment")) -> str:
    (tmp_path / "frozen_config.yaml").write_text("statistics: {alpha: 0.05}\n", encoding="utf-8")
    for stage in completed:
        _write(tmp_path / "stage_status" / f"{stage}.json", {"status": "completed"})
    _write(tmp_path / "statistics" / "quantum_scaling_statistics.json", dict(
        primary_amplification=dict(mean_log10_amplification=0.8, ci_low=0.3, ci_high=1.2, n_clusters=12,
                                   p_gatekeeping_adjusted=primary_p),
        primary=dict(mean_slope=-0.1, ci_low=-0.3, ci_high=0.1, n_clusters=12, p_gatekeeping_adjusted=slope_p)))
    _write(tmp_path / "qc_benchmark" / "statistics_outputs.json", dict(effects=[dict(
        baseline="sa", metric="hit", mean_difference=-0.2, n_clusters=12, gatekeeping_family="secondary",
        p_gatekeeping_adjusted=None)]))
    _write(tmp_path / "statistics" / "structure_statistics.json", dict(
        primary=dict(endpoint="final_rmsd", mean_difference=0.05, ci_low=-0.1, ci_high=0.2, n_clusters=11,
                     p_holm_confirmatory_family=0.6),
        rq5=dict(spearman_rho=0.1, ci_low=-0.2, ci_high=0.4, p_holm_confirmatory_family=0.6)))
    return "\n".join(section_research_question_ledger(ReportContext(tmp_path)))


def test_closed_gatekeeping_makes_classical_comparisons_descriptive(tmp_path) -> None:
    text = _run(tmp_path, primary_p=0.01, slope_p=0.4)
    assert "QAOA concentrates probability on the ground state" in text      # Q2 rejected, positive
    assert "| Q3 Scaling slope" in text and "null not rejected (adjusted p=0.4" in text
    assert "serial gatekeeping keeps every QAOA-vs-classical comparison closed" in text
    assert "QAOA vs sa" in text and "not testable" in text
    assert "Q5 Structural endpoint" in text


def test_missing_stages_are_not_established(tmp_path) -> None:
    text = _run(tmp_path, primary_p=0.01, slope_p=0.01, completed=("qc_benchmark",))
    assert text.count("not established (evidence stage incomplete)") >= 3
    assert "Q5 is not established" in text
