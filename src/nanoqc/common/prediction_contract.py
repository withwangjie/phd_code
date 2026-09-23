"""Fail-closed metadata gate separating retrospective controls from prediction.

This validates declarations and audit evidence linkage; it does not infer family
independence from a user assertion or replace independent sequence/structure audits.
"""
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json


def validate_prediction_contract(case: Mapping[str, Any], manifest_directory: Path) -> None:
    """Require explicit non-reference inputs, degrees of freedom and split audit."""
    if case.get('protocol') != 'prediction':
        return
    required = ('input_origin', 'reference_used_for_input', 'reference_used_for_active_selection',
                'degrees_of_freedom', 'independence_audit', 'independence_audit_sha256')
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f'Prediction requires explicit provenance: {missing}')
    if case['input_origin'] not in ('external_predicted_complex', 'unbound_docking_pose'):
        raise ValueError('Native/perturbed bound inputs must be validation_control')
    if case['reference_used_for_input'] is not False or case['reference_used_for_active_selection'] is not False:
        raise ValueError('Prediction cannot use the reference for inputs or active selection')
    if case['degrees_of_freedom'] != 'fixed_backbone_sidechains':
        raise ValueError('Current all-atom engine supports fixed_backbone_sidechains only')
    root = Path(manifest_directory).resolve()

    def confined(relative: Any, label: str) -> Path:
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f'{label} must be a relative path inside the manifest directory')
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f'{label} escapes the manifest directory')
        return candidate

    path = confined(case['independence_audit'], 'independence_audit')
    if not path.is_file():
        raise ValueError('independence_audit file does not exist')
    if hashlib.sha256(path.read_bytes()).hexdigest() != case['independence_audit_sha256']:
        raise ValueError('Independence audit hash mismatch')
    audit = json.loads(path.read_text(encoding='utf-8'))
    if audit.get('status') != 'passed' or audit.get('used_for_development') is not False:
        raise ValueError('Prediction requires a passed, unused-for-development split audit')
    if not audit.get('target_id') or audit.get('target_id') != case.get('target_id'):
        raise ValueError('Split audit must identify the same target')
    if audit.get('training_pdb_overlap') is not False or audit.get('family_overlap') is not False:
        raise ValueError('PDB and family overlap audits must explicitly pass')
    if not audit.get('method') or not audit.get('evidence_files'):
        raise ValueError('Split audit must include method and hashed evidence files')
    for relative, digest in audit['evidence_files'].items():
        evidence = confined(relative, f'evidence file {relative!r}')
        if not evidence.is_file():
            raise ValueError(f'Split evidence file does not exist: {relative}')
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Split evidence changed: {relative}')
