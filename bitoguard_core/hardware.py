from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _detect_gpus() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


@dataclass(frozen=True)
class HardwareProfile:
    cpu_threads: int
    gpu_available: bool
    gpu_enabled: bool
    gpu_count: int
    gpu_device_id: int | None
    gpu_name: str | None
    fold_workers: int
    worker_cpu_threads: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def hardware_profile() -> HardwareProfile:
    cpu_threads = _env_int("BITOGUARD_CPU_THREADS", os.cpu_count() or 1)
    gpu_names = _detect_gpus()
    gpu_available = bool(gpu_names)
    gpu_count = len(gpu_names)

    force_gpu = os.getenv("BITOGUARD_USE_GPU", "").strip().lower()
    if force_gpu in {"0", "false", "no", "off"}:
        gpu_enabled = False
    elif force_gpu in {"1", "true", "yes", "on"}:
        gpu_enabled = gpu_available
    else:
        gpu_enabled = gpu_available

    requested_gpu = _env_int("BITOGUARD_GPU_DEVICE_ID", 0)
    gpu_device_id = None
    gpu_name = None
    if gpu_enabled and gpu_count > 0:
        gpu_device_id = min(requested_gpu, gpu_count - 1)
        gpu_name = gpu_names[gpu_device_id]

    default_fold_workers = 1 if gpu_enabled else min(cpu_threads, max(1, cpu_threads // 2))
    fold_workers = _env_int("BITOGUARD_FOLD_WORKERS", default_fold_workers)
    fold_workers = max(1, min(fold_workers, cpu_threads))
    worker_cpu_threads = max(1, cpu_threads // fold_workers)

    return HardwareProfile(
        cpu_threads=cpu_threads,
        gpu_available=gpu_available,
        gpu_enabled=gpu_enabled,
        gpu_count=gpu_count,
        gpu_device_id=gpu_device_id,
        gpu_name=gpu_name,
        fold_workers=fold_workers,
        worker_cpu_threads=worker_cpu_threads,
    )


def describe_hardware() -> str:
    profile = hardware_profile()
    if profile.gpu_enabled and profile.gpu_name:
        return (
            f"cpu_threads={profile.cpu_threads}, gpu={profile.gpu_name}, "
            f"gpu_device_id={profile.gpu_device_id}, fold_workers={profile.fold_workers}, "
            f"worker_cpu_threads={profile.worker_cpu_threads}"
        )
    return (
        f"cpu_threads={profile.cpu_threads}, gpu=disabled, "
        f"fold_workers={profile.fold_workers}, worker_cpu_threads={profile.worker_cpu_threads}"
    )


def lightgbm_runtime_params() -> dict[str, Any]:
    profile = hardware_profile()
    if profile.gpu_enabled:
        return {
            "device_type": "gpu",
            "max_bin": 255,
            # Half CPU threads in GPU path — leaves headroom for concurrent CatBoost CPU Base B.
            "n_jobs": max(1, profile.cpu_threads // 2),
        }
    return {"n_jobs": profile.cpu_threads}


def xgboost_runtime_params() -> dict[str, Any]:
    profile = hardware_profile()
    if profile.gpu_enabled:
        # In GPU path, XGBoost uses GPU for tree building; n_jobs only affects
        # preprocessing. Omit to leave CPU threads free for concurrent Base B.
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist", "device": "cpu", "n_jobs": profile.cpu_threads}


def catboost_runtime_params() -> dict[str, Any]:
    profile = hardware_profile()
    params: dict[str, Any] = {
        "thread_count": profile.cpu_threads,
    }
    # BITOGUARD_CATBOOST_CPU_ONLY=1 forces CPU mode even when GPU is available.
    # Useful for running GPU-accelerated GraphSAGE alongside CPU CatBoost.
    catboost_cpu_only = os.getenv("BITOGUARD_CATBOOST_CPU_ONLY", "0").strip() == "1"
    if profile.gpu_enabled and not catboost_cpu_only:
        params["task_type"] = "GPU"
        if profile.gpu_device_id is not None:
            params["devices"] = str(profile.gpu_device_id)
        # Safety margin for coexisting Jupyter kernels using VRAM.
        params["gpu_ram_part"] = 0.5
        # Plain boosting is faster on GPU (no ordering overhead vs Ordered default).
        params["boosting_type"] = "Plain"
    else:
        params["task_type"] = "CPU"
    return params


def sklearn_n_jobs() -> int:
    return hardware_profile().cpu_threads


def fold_worker_count() -> int:
    return hardware_profile().fold_workers


def fold_worker_env() -> dict[str, str]:
    profile = hardware_profile()
    env = os.environ.copy()
    env["BITOGUARD_CPU_THREADS"] = str(profile.worker_cpu_threads)
    return env
