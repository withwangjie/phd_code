"""Shared readers, formatting helpers and the ReportContext for the final report.

Split out of generate_final_research_report.py, which re-exports every name here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"PyYAML is required to read frozen_config.yaml: {exc}")


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _filter_qc_rows(rows: List[Dict[str, str]], budget_mode: str) -> List[Dict[str, str]]:
    selected=[]
    for row in rows:
        mode=row.get("budget_mode") or "matched_outputs"
        if mode==budget_mode:
            selected.append(row)
    return selected


def _formal_statistics_payload(ctx: "ReportContext", mode: str) -> Optional[Dict[str, Any]]:
    payload=_read_json(ctx.run_dir / "qc_benchmark" / f"statistics_{mode}.json")
    return payload if isinstance(payload,dict) else None


def _paired_inference_multiplicity_note(mode: str) -> str:
    """Return protocol-faithful multiplicity language for a budget mode."""
    if mode=="outputs":
        return (
            "Multiplicity: serial gatekeeping. The quantum-intrinsic primary family "
            "(primary-size exact ground-state amplification and its scaling slope) is "
            "Holm-adjusted alone; matched-output QAOA-vs-classical effects are secondary "
            "and can be interpreted inferentially only after both primary hypotheses are "
            "rejected, except abstract shot/query QTS99 effects, which are descriptive only. "
            "The all-effects Holm column is descriptive."
        )
    if mode=="time":
        return (
            "Matched-time analyses are descriptive controls for classical emulation cost "
            "and are outside the confirmatory serial-gatekeeping family. Their raw p values "
            "and all-effects Holm values are descriptive and must not be interpreted as "
            "gatekeeping-adjusted confirmatory evidence."
        )
    raise ValueError(f"Unsupported paired-statistics budget mode: {mode}")


def _primary_active_sites(ctx: "ReportContext") -> int:
    return int(((ctx.frozen_config.get("statistics",{}) or {}).get("primary_active_sites",6)))


def _primary_pruning(ctx: "ReportContext") -> str:
    return str(((ctx.frozen_config.get("statistics",{}) or {}).get("primary_pruning","egnn")))


def _quantum_primary(ctx: "ReportContext") -> Dict[str, Any]:
    return dict((((ctx.frozen_config.get("quantum_protocol",{}) or {}).get("primary",{}) or {})))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


# ---------------------------------------------------------------------------
# Report context
# ---------------------------------------------------------------------------

class ReportContext:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.frozen_config = yaml.safe_load(_read_text(run_dir / "frozen_config.yaml") or "{}") or {}
        self.run_manifest = _read_json(run_dir / "run_manifest.json") or {}
        self.seed_streams = _read_json(run_dir / "seed_streams.json") or {}
        self.progress = _read_json(run_dir / "progress.json") or {}
        self.stage_status = {
            path.stem: _read_json(path)
            for path in sorted((run_dir / "stage_status").glob("*.json"))
        } if (run_dir / "stage_status").is_dir() else {}
        self.repo_root = self._resolve_repo_root()

    def _resolve_repo_root(self) -> Path:
        raw = (self.frozen_config.get("paths", {}) or {}).get("repo_root", ".")
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (REPO_ROOT).resolve()

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.repo_root / relative)

    def resolve_run(self, relative: str) -> Path:
        """Resolve a path relative to THIS run's own run_dir, not repo_root.

        paths.dataset_dir/paths.checkpoint_dir are directory NAMES scoped
        under run_dir (run isolation -- see run_full_experiment.py's
        dataset_dir()/checkpoint_dir()), never a fixed repo-root path shared
        across runs, so they must be resolved here the same way."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.run_dir / relative)


# ---------------------------------------------------------------------------
# Section 0: stage completion table
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "env_check", "smoke_check", "data_audit", "queue_freeze", "egnn_train",
    "energy_calibration", "method_sensitivity", "qc_benchmark",
    "structure_experiment", "external_validation", "statistics", "final_report",
]


def stage_ok(ctx: ReportContext, stage: str) -> bool:
    record = ctx.stage_status.get(stage)
    return bool(record and record["status"] in ("completed", "completed_with_failures"))
