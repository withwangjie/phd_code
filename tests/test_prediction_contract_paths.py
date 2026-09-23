from __future__ import annotations

import hashlib
import json

import pytest

from nanoqc.common.prediction_contract import validate_prediction_contract


def _case(audit_name: str, evidence_name: str) -> dict:
    return {
        "protocol": "prediction",
        "input_origin": "external_predicted_complex",
        "reference_used_for_input": False,
        "reference_used_for_active_selection": False,
        "degrees_of_freedom": "fixed_backbone_sidechains",
        "independence_audit": audit_name,
        "independence_audit_sha256": "placeholder",
        "target_id": "1abc",
        "_evidence_name": evidence_name,
    }


def test_prediction_contract_rejects_audit_path_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside_audit.json"
    outside.write_text("{}", encoding="utf-8")
    case = _case("../outside_audit.json", "evidence.json")
    with pytest.raises(ValueError, match="escapes|relative path"):
        validate_prediction_contract(case, tmp_path)


def test_prediction_contract_rejects_evidence_path_escape(tmp_path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("evidence", encoding="utf-8")
    audit = tmp_path / "audit.json"
    payload = {
        "status": "passed",
        "used_for_development": False,
        "target_id": "1abc",
        "training_pdb_overlap": False,
        "family_overlap": False,
        "method": "test",
        "evidence_files": {"../outside_evidence.json": "0" * 64},
    }
    audit.write_text(json.dumps(payload), encoding="utf-8")
    case = _case("audit.json", "evidence.json")
    case["independence_audit_sha256"] = hashlib.sha256(audit.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="escapes|relative path"):
        validate_prediction_contract(case, tmp_path)
