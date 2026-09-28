from nanoqc.pipeline.resolve_server_config import _find_existing


def test_find_existing_accepts_absolute_directory_path(tmp_path):
    assert _find_existing([tmp_path]) == str(tmp_path.resolve())


def test_80_core_server_cpu_budget_and_graph_cap(tmp_path, monkeypatch):
    from pathlib import Path
    import yaml
    from nanoqc.pipeline import resolve_server_config as resolver

    server=yaml.safe_load((Path(__file__).resolve().parents[1]/
                           "configs/server_config.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(resolver,"_cpu_count",lambda:80)
    monkeypatch.setattr(resolver,"_ram_gb",lambda:256.)
    monkeypatch.setattr(resolver,"_gpu_count",lambda:2)
    monkeypatch.setattr(resolver,"_resolve_data_root",lambda server:str(tmp_path))
    monkeypatch.setattr(resolver,"_resolve_run_root",lambda server,root:str(tmp_path/"runs"))
    monkeypatch.setattr(resolver,"_resolve_venv",lambda server:str(tmp_path/"venv"))
    monkeypatch.setattr(resolver,"_resolve_tool",lambda *args:None)
    config,report=resolver.resolve({},server)
    assert config["qc_benchmark"]["workers"]==64
    assert config["data_audit"]["workers"]==64
    assert config["queue_freeze"]["graph_build"]["workers"]==16
    assert config["egnn_train"]["nproc_per_node"]==2
    assert config["hardware"]["cpu_threads_per_process"]==1
    assert config["hardware"]["structural_gpu_devices"]==["0","1"]
    assert report["external_audit_workers"]==12
    server["resources"]["cpu_threads_per_process"]=4
    config,_=resolver.resolve({},server)
    assert config["qc_benchmark"]["workers"]==18
    assert 18*config["hardware"]["cpu_threads_per_process"]<=72
