from nanoqc.pipeline.resolve_server_config import _find_existing


def test_find_existing_accepts_absolute_directory_path(tmp_path):
    assert _find_existing([tmp_path]) == str(tmp_path.resolve())
