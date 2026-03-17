from __future__ import annotations

import hardware


def test_hardware_profile_prefers_gpu_when_available(monkeypatch) -> None:
    hardware.hardware_profile.cache_clear()
    monkeypatch.setattr(hardware, "_detect_gpus", lambda: ["GPU 0"])
    monkeypatch.delenv("BITOGUARD_USE_GPU", raising=False)
    monkeypatch.setenv("BITOGUARD_CPU_THREADS", "8")
    monkeypatch.delenv("BITOGUARD_FOLD_WORKERS", raising=False)

    profile = hardware.hardware_profile()

    assert profile.gpu_available is True
    assert profile.gpu_enabled is True
    assert profile.gpu_name == "GPU 0"
    assert profile.cpu_threads == 8
    assert profile.fold_workers == 1
    assert profile.worker_cpu_threads == 8


def test_hardware_profile_honors_cpu_only_override(monkeypatch) -> None:
    hardware.hardware_profile.cache_clear()
    monkeypatch.setattr(hardware, "_detect_gpus", lambda: ["GPU 0"])
    monkeypatch.setenv("BITOGUARD_USE_GPU", "0")
    monkeypatch.setenv("BITOGUARD_CPU_THREADS", "8")
    monkeypatch.setenv("BITOGUARD_FOLD_WORKERS", "4")

    profile = hardware.hardware_profile()

    assert profile.gpu_available is True
    assert profile.gpu_enabled is False
    assert profile.fold_workers == 4
    assert profile.worker_cpu_threads == 2


def test_runtime_param_helpers_reflect_profile(monkeypatch) -> None:
    hardware.hardware_profile.cache_clear()
    monkeypatch.setattr(hardware, "_detect_gpus", lambda: ["GPU 0"])
    monkeypatch.setenv("BITOGUARD_USE_GPU", "1")
    monkeypatch.setenv("BITOGUARD_CPU_THREADS", "6")
    monkeypatch.setenv("BITOGUARD_CATBOOST_CPU_ONLY", "0")

    assert hardware.lightgbm_runtime_params()["device_type"] == "gpu"
    assert hardware.lightgbm_runtime_params()["n_jobs"] == 6
    assert hardware.xgboost_runtime_params()["device"] == "cuda"
    assert hardware.xgboost_runtime_params()["n_jobs"] == 6
    assert hardware.catboost_runtime_params()["task_type"] == "GPU"
    assert hardware.catboost_runtime_params()["thread_count"] == 6

    hardware.hardware_profile.cache_clear()
