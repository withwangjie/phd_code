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
    monkeypatch.setattr(resolver,"_logical_cpu_count",lambda:80)
    monkeypatch.setattr(resolver,"_ram_gb",lambda:256.)
    monkeypatch.setattr(resolver,"_gpu_count",lambda:2)
    monkeypatch.setattr(resolver,"_resolve_data_root",lambda server:str(tmp_path))
    monkeypatch.setattr(resolver,"_resolve_run_root",lambda server,root:str(tmp_path/"runs"))
    monkeypatch.setattr(resolver,"_resolve_venv",lambda server:str(tmp_path/"venv"))
    monkeypatch.setattr(resolver,"_resolve_tool",lambda *args:None)
    config,report=resolver.resolve({},server)
    assert config["qc_benchmark"]["workers"]==64
    assert config["data_audit"]["workers"]==64
    assert report["cpu_min_workers_target"]==48
    assert report["cpu_max_workers"]==64
    assert report["cpu_min_workers_target_met"] is True
    assert report["cpu_pool_utilization"]==0.8
    assert config["hardware"]["cpu_monitor_enabled"] is True
    assert config["queue_freeze"]["graph_build"]["workers"]==44  # RAM-bounded, not 16 threads (A35)
    assert report["graph_build_workers_ram_limited"] is True
    assert config["egnn_train"]["nproc_per_node"]==2
    assert config["hardware"]["cpu_threads_per_process"]==1
    assert config["hardware"]["structural_gpu_devices"]==["0","1"]
    assert config["hardware"]["structural_workers_per_gpu"]==4
    assert config["hardware"]["structural_target_workers"]==8
    assert config["hardware"]["structural_prepare_workers"]==16  # 8 per device (A35)
    assert config["hardware"]["structural_prepare_workers_per_gpu"]==8
    assert config["hardware"]["calibration_workers"]==8
    assert config["hardware"]["calibration_workers_per_gpu"]==4
    assert config["hardware"]["gpu_monitor_enabled"] is True
    assert report["structural_workers_per_gpu"]==4
    assert report["external_audit_workers"]==64  # follows the CPU pool (A36)
    assert report["foldseek_prepare_workers"]==64
    assert report["egnn_replicate_workers"]==2
    server["resources"]["cpu_threads_per_process"]=4
    config,report=resolver.resolve({},server)
    assert config["qc_benchmark"]["workers"]==16
    assert 16*config["hardware"]["cpu_threads_per_process"]<=0.8*80
    assert report["cpu_min_workers_target_met"] is False
    assert report["cpu_worker_limit_reason"]=="available cores / threads per process / utilization target"
    server["resources"]["max_workers"]=32
    import pytest
    with pytest.raises(SystemExit,match="min_workers <= max_workers"):
        resolver.resolve({},server)


def _resolve_on(tmp_path, monkeypatch, *, physical, logical, ram=256., **resources):
    from pathlib import Path
    import yaml
    from nanoqc.pipeline import resolve_server_config as resolver

    server=yaml.safe_load((Path(__file__).resolve().parents[1]/
                           "configs/server_config.yaml").read_text(encoding="utf-8"))
    server["resources"].update(resources)
    monkeypatch.setattr(resolver,"_cpu_count",lambda:physical)
    monkeypatch.setattr(resolver,"_logical_cpu_count",lambda:logical)
    monkeypatch.setattr(resolver,"_ram_gb",lambda:ram)
    monkeypatch.setattr(resolver,"_gpu_count",lambda:2)
    monkeypatch.setattr(resolver,"_resolve_data_root",lambda server:str(tmp_path))
    monkeypatch.setattr(resolver,"_resolve_run_root",lambda server,root:str(tmp_path/"runs"))
    monkeypatch.setattr(resolver,"_resolve_venv",lambda server:str(tmp_path/"venv"))
    monkeypatch.setattr(resolver,"_resolve_tool",lambda *args:None)
    return resolver.resolve({},server)


def test_hyperthreaded_80_vcpu_host_reaches_the_utilization_target(tmp_path, monkeypatch):
    # 40 physical cores x 2 threads: physical counting would give 32 workers (40 %).
    config,report=_resolve_on(tmp_path,monkeypatch,physical=40,logical=80)
    assert config["qc_benchmark"]["workers"]==64
    assert report["cpu_count_basis"]=="logical" and report["cpu_counted"]==80
    assert report["cpu_physical_cores"]==40
    assert report["cpu_pool_utilization"]>=0.8
    config,report=_resolve_on(tmp_path,monkeypatch,physical=40,logical=80,cpu_count_basis="physical")
    assert config["qc_benchmark"]["workers"]==32
    assert report["cpu_min_workers_target_met"] is False


def test_auto_pool_scales_with_logical_cpus_and_respects_ram(tmp_path, monkeypatch):
    config,report=_resolve_on(tmp_path,monkeypatch,physical=80,logical=160)
    assert config["qc_benchmark"]["workers"]==128 and report["cpu_pool_utilization"]==0.8
    config,report=_resolve_on(tmp_path,monkeypatch,physical=80,logical=160,ram=64.)
    # (64 GB - 32 GB free-RAM floor) / 1.5 GB per worker
    assert config["qc_benchmark"]["workers"]==21
    assert report["cpu_worker_limit_reason"]=="RAM per worker"


def test_auto_pool_requires_a_valid_target(tmp_path, monkeypatch):
    import pytest
    with pytest.raises(SystemExit,match="requires target_cpu_utilization"):
        _resolve_on(tmp_path,monkeypatch,physical=80,logical=80,target_cpu_utilization=None)
    with pytest.raises(SystemExit,match=r"target_cpu_utilization must be in \(0, 1\]"):
        _resolve_on(tmp_path,monkeypatch,physical=80,logical=80,target_cpu_utilization=1.5)
    with pytest.raises(SystemExit,match="cpu_count_basis"):
        _resolve_on(tmp_path,monkeypatch,physical=80,logical=80,cpu_count_basis="sockets")


def test_graph_workers_follow_ram_and_reject_auto_without_an_estimate(tmp_path, monkeypatch):
    import pytest
    config,report=_resolve_on(tmp_path,monkeypatch,physical=80,logical=80,ram=512.)
    # (512 GB - 32 GB floor) / 5 GB exceeds the 64-worker pool, so the pool bounds it.
    assert config["queue_freeze"]["graph_build"]["workers"]==64
    assert report["graph_build_workers_ram_limited"] is False
    config,_=_resolve_on(tmp_path,monkeypatch,physical=80,logical=80,max_graph_workers=8)
    assert config["queue_freeze"]["graph_build"]["workers"]==8
    with pytest.raises(SystemExit,match="requires graph_worker_ram_gb"):
        _resolve_on(tmp_path,monkeypatch,physical=80,logical=80,graph_worker_ram_gb=None)


def test_cpu_pool_followers_accept_an_explicit_cap(tmp_path, monkeypatch):
    config,report=_resolve_on(tmp_path,monkeypatch,physical=80,logical=80,
                              max_foldseek_prepare_workers=6,max_external_audit_workers=6)
    assert report["foldseek_prepare_workers"]==6
    assert report["external_audit_workers"]==6
    assert config["hardware"]["foldseek_prepare_workers"]==6
