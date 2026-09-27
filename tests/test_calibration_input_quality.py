"""Only missing observed heavy atoms may leave the calibration denominator."""

from nanoqc.experiments.generate_energy_calibration_dataset import is_input_quality_exclusion


def test_missing_heavy_atoms_are_input_quality_exclusions():
    assert is_input_quality_exclusion(ValueError("Missing heavy atoms H:114: ['CG']"))
    assert is_input_quality_exclusion(ValueError("Internal heavy-atom repair would be required"))


def test_scientific_and_runtime_failures_remain_generation_failures():
    assert not is_input_quality_exclusion(ValueError("Raw/graph protein residue identities differ"))
    assert not is_input_quality_exclusion(ValueError("Fixed three-state scaling requires a candidate in each chi1 well"))
    assert not is_input_quality_exclusion(RuntimeError("OpenMM CUDA failure"))
