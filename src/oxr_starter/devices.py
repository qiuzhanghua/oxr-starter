"""设备抽象：CUDA / MPS / CPU。

原项目只支持 NVIDIA（`torch.cuda` + `device_map="cuda:0"` + `bfloat16`）。这里把
"选设备、选 dtype、报显存"收敛到一处，**原有 CUDA 行为完全不变**：

| 设备 | 写法 | dtype |
|---|---|---|
| CUDA | `cuda:0` + `device_map="cuda:0"` | `bfloat16`（与原实现一致）|
| MPS  | `mps` + `.to("mps")` | `float16`（实测比 bf16 快约 16%，输出一致）|
| CPU  | `cpu` | `float32`（兜底，仅用于冒烟测试）|

MPS 没有 `max_memory_allocated()`，峰值靠 `PeakTracker` 轮询
`driver_allocated_memory()` 采样；CUDA 仍用官方统计。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Self

import torch

CUDA = "cuda"
MPS = "mps"
CPU = "cpu"

_DTYPE_ALIASES = {
    "auto": None,
    "": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


class DeviceError(RuntimeError):
    """本机没有可用推理设备，或参数非法。"""


def gb(n: float) -> float:
    return round(n / 1024**3, 2)


def mps_available() -> bool:
    try:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except AttributeError:  # 老版本 torch 没有 mps 命名空间
        return False


def cuda_available() -> bool:
    return torch.cuda.is_available()


def available_devices() -> list[str]:
    """按优先级返回本机可用设备。"""
    out: list[str] = []
    if cuda_available():
        out.append(CUDA)
    if mps_available():
        out.append(MPS)
    out.append(CPU)
    return out


def pick_device(prefer: str | None = None) -> str:
    """选设备。`prefer` 为空/auto 时按 cuda > mps > cpu 自动挑。"""
    if prefer in (None, "", "auto"):
        return available_devices()[0]
    name = prefer.strip().lower()
    if name.startswith("cuda"):
        if not cuda_available():
            raise DeviceError(
                f"指定了 {prefer!r}，但本机 CUDA 不可用（可用：{available_devices()}）"
            )
        return CUDA
    if name == MPS:
        if not mps_available():
            raise DeviceError(
                "指定了 mps，但 torch.backends.mps 不可用"
                "（需要 Apple Silicon + macOS 12.3+，且 torch 为 arm64 构建）"
            )
        return MPS
    if name == CPU:
        return CPU
    raise DeviceError(f"未知设备 {prefer!r}（可选：auto / cuda / mps / cpu）")


def default_dtype(device: str) -> torch.dtype:
    return {CUDA: torch.bfloat16, MPS: torch.float16, CPU: torch.float32}[device]


def resolve_dtype(device: str, name: str | None = None) -> torch.dtype:
    """把 `--dtype` 解析成 torch.dtype；None/auto 时用该设备的默认值。"""
    key = "auto" if name is None else str(name).strip().lower()
    if key not in _DTYPE_ALIASES:
        raise DeviceError(f"未知 dtype {name!r}（可选：auto / bf16 / fp16 / fp32）")
    dtype = _DTYPE_ALIASES[key]
    return default_dtype(device) if dtype is None else dtype


def torch_device(device: str) -> torch.device:
    return torch.device("cuda:0" if device == CUDA else device)


def device_kwargs(device: str) -> dict:
    """`from_pretrained` 的设备参数：CUDA 保持原写法，其余在加载后 `.to()`。"""
    if device == CUDA:
        return {"device_map": "cuda:0"}
    return {}


def prepare(device: str) -> None:
    """按设备做一次性设置（复位统计 / 可选显存上限）。"""
    if device == CUDA:
        torch.cuda.reset_peak_memory_stats()
    elif device == MPS:
        # 长页 + 大 max_tokens 时防止吃掉整机内存；默认不限制（沿用 torch 默认）
        frac = os.environ.get("OXR_MPS_MEM_FRACTION")
        if frac:
            torch.mps.set_per_process_memory_fraction(float(frac))


def move_model(model, device: str):
    """CUDA 由 device_map 放置；MPS/CPU 显式搬过去。"""
    if device == CUDA:
        return model
    return model.to(torch_device(device))


def to_device(x, dev: torch.device, dtype: torch.dtype | None = None):
    """把 preprocess 的输出搬到设备上（tuple/list 先 cat，与上游写法一致）。"""
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        x = torch.cat(list(x), dim=0)
    return x.to(dev) if dtype is None else x.to(device=dev, dtype=dtype)


def allocated_gb(device: str) -> float:
    if device == CUDA:
        return gb(torch.cuda.memory_allocated())
    if device == MPS:
        return gb(torch.mps.current_allocated_memory())
    return 0.0


def driver_gb(device: str) -> float:
    """进程在驱动层占用的显存（MPS 只有这个口径）。"""
    if device == CUDA:
        return gb(torch.cuda.memory_reserved())
    if device == MPS:
        return gb(torch.mps.driver_allocated_memory())
    return 0.0


def peak_gb(device: str) -> float:
    if device == CUDA:
        return gb(torch.cuda.max_memory_allocated())
    if device == MPS:
        return gb(torch.mps.driver_allocated_memory())
    return 0.0


def empty_cache(device: str) -> None:
    if device == CUDA:
        torch.cuda.empty_cache()
    elif device == MPS:
        torch.mps.empty_cache()


class PeakTracker:
    """MPS 没有峰值统计 API，用后台轮询采样（CUDA 直接读官方计数）。

    用法：
        with PeakTracker(dev) as peak:
            ...
            print(peak.peak_gb)
    """

    def __init__(self, device: str, interval: float = 0.05):
        self.device = device
        self.interval = interval
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cuda = device == CUDA

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.device == MPS:
                self.peak_gb = max(
                    self.peak_gb, gb(torch.mps.driver_allocated_memory())
                )
            time.sleep(self.interval)

    def __enter__(self) -> Self:
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()
        else:
            self.peak_gb = driver_gb(self.device)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._cuda:
            self.peak_gb = peak_gb(self.device)
        else:
            self.peak_gb = max(self.peak_gb, driver_gb(self.device))
        # 停止后统一走 driver 口径，避免与采样值口径不一致
        if self.device != CUDA:
            self.peak_gb = round(self.peak_gb, 2)


def describe(device: str, dtype: torch.dtype | None = None) -> str:
    """一行人类可读的设备描述。"""
    dt = f", {str(dtype).replace('torch.', '')}" if dtype is not None else ""
    if device == CUDA:
        p = torch.cuda.get_device_properties(0)
        return f"CUDA {p.name}（总显存 {gb(p.total_memory)} GB{dt}）"
    if device == MPS:
        return f"Apple MPS（统一内存 {gb(torch.mps.recommended_max_memory())} GB{dt}）"
    return f"CPU{dt}"


def memory_line(device: str) -> str:
    """`驱动占用 / 分配` 的简写，用于日志（峰值另见 PeakTracker）。"""
    if device == CPU:
        return "CPU：无显存统计"
    return f"驱动占用 {driver_gb(device)} GB / 分配 {allocated_gb(device)} GB"
