"""A scientific failure must still leave an auditable run report."""

from nanoqc.pipeline.orchestrator_common import StageResult
from nanoqc.pipeline.run_full_experiment import write_incomplete_research_report
from nanoqc.reporting import generate_final_research_report


def test_failed_run_writes_diagnostic_report_without_changing_status(tmp_path, monkeypatch):
    failure = StageResult("structure_experiment", "failed", "start", "end", 1,
                          "14 controls failed physical acceptance")
    results = {"structure_experiment": failure}
    monkeypatch.setattr(generate_final_research_report, "compile_report",
                        lambda run_dir: "## Available quantum results\nQC cases: 18800")

    path = write_incomplete_research_report(tmp_path, results)

    text = path.read_text(encoding="utf-8")
    assert path.name == "INCOMPLETE_RESEARCH_REPORT.md"
    assert "did not satisfy every formal acceptance gate" in text
    assert "QC cases: 18800" in text
    assert failure.status == "failed"


def test_report_renderer_failure_still_preserves_stage_failure(tmp_path, monkeypatch):
    failure = StageResult("structure_experiment", "failed", "start", "end", 1,
                          "validation queue failed physical acceptance")

    def broken_renderer(run_dir):
        raise ValueError("missing partial statistics")

    monkeypatch.setattr(generate_final_research_report, "compile_report", broken_renderer)
    path = write_incomplete_research_report(tmp_path, {"structure_experiment": failure})
    text = path.read_text(encoding="utf-8")
    assert "Detailed report rendering failed" in text
    assert "structure_experiment: failed" in text
    assert "validation queue failed physical acceptance" in text
