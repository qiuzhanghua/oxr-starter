"""devices.py：设备选择、dtype、显存口径与 PeakTracker。

不依赖真实 GPU：CUDA / MPS 的分支用 monkeypatch 替换 torch 的统计接口，
所以同一套用例在 Linux CPU 机器上也能跑。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from oxr_starter import devices as d


def test_gb_rounds_to_two_decimals():
    assert d.gb(1024**3) == 1.0
    assert d.gb(1536 * 1024**2) == 1.5
    assert d.gb(0) == 0.0


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, ["cuda", "mps", "cpu"]),
        (True, False, ["cuda", "cpu"]),
        (False, True, ["mps", "cpu"]),
        (False, False, ["cpu"]),
    ],
)
def test_available_devices_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(d, "cuda_available", lambda: cuda)
    monkeypatch.setattr(d, "mps_available", lambda: mps)

    assert d.available_devices() == expected
    assert d.pick_device("auto") == expected[0]
    assert d.pick_device(None) == expected[0]
    assert d.pick_device("") == expected[0]


def test_pick_device_accepts_cuda_index(monkeypatch):
    monkeypatch.setattr(d, "cuda_available", lambda: True)
    assert d.pick_device("cuda:1") == "cuda"
    assert d.pick_device(" CUDA ") == "cuda"


def test_pick_device_strips_mps(monkeypatch):
    monkeypatch.setattr(d, "mps_available", lambda: True)
    assert d.pick_device(" MPS ") == "mps"
    assert d.pick_device("cpu") == "cpu"


def test_pick_device_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(d, "cuda_available", lambda: False)
    monkeypatch.setattr(d, "mps_available", lambda: False)
    with pytest.raises(d.DeviceError, match="CUDA 不可用"):
        d.pick_device("cuda")


def test_pick_device_mps_unavailable(monkeypatch):
    monkeypatch.setattr(d, "mps_available", lambda: False)
    with pytest.raises(d.DeviceError, match="torch.backends.mps 不可用"):
        d.pick_device("mps")


def test_pick_device_unknown_name():
    with pytest.raises(d.DeviceError, match="未知设备"):
        d.pick_device("tpu")


def test_mps_available_falls_back_when_torch_too_old(monkeypatch):
    monkeypatch.setattr(d.torch, "backends", SimpleNamespace())
    assert d.mps_available() is False


def test_cuda_available_delegates_to_torch(monkeypatch):
    monkeypatch.setattr(d.torch.cuda, "is_available", lambda: True)
    assert d.cuda_available() is True


@pytest.mark.parametrize(
    ("device", "name", "expected"),
    [
        ("cuda", None, torch.bfloat16),
        ("cuda", "auto", torch.bfloat16),
        ("cuda", "", torch.bfloat16),
        ("mps", None, torch.float16),
        ("mps", "bf16", torch.bfloat16),
        ("mps", "bfloat16", torch.bfloat16),
        ("cpu", None, torch.float32),
        ("cpu", "FP16", torch.float16),
        ("cpu", " half ", torch.float16),
        ("cpu", "float32", torch.float32),
    ],
)
def test_resolve_dtype_aliases(device, name, expected):
    assert d.resolve_dtype(device, name) is expected


def test_resolve_dtype_rejects_unknown():
    with pytest.raises(d.DeviceError, match="未知 dtype"):
        d.resolve_dtype("cpu", "int8")


def test_torch_device_maps_cuda_to_zero():
    assert d.torch_device("cuda") == torch.device("cuda:0")
    assert d.torch_device("mps") == torch.device("mps")
    assert d.torch_device("cpu") == torch.device("cpu")


def test_device_kwargs_only_cuda_uses_device_map():
    assert d.device_kwargs("cuda") == {"device_map": "cuda:0"}
    assert d.device_kwargs("mps") == {}
    assert d.device_kwargs("cpu") == {}


def test_prepare_cuda_resets_peak_stats(monkeypatch):
    calls = []
    monkeypatch.setattr(d.torch.cuda, "reset_peak_memory_stats", lambda: calls.append(1))
    d.prepare("cuda")
    assert calls == [1]


def test_prepare_mps_sets_memory_fraction_when_configured(monkeypatch):
    calls = []
    monkeypatch.setattr(
        d.torch,
        "mps",
        SimpleNamespace(set_per_process_memory_fraction=lambda f: calls.append(f)),
    )
    monkeypatch.setenv("OXR_MPS_MEM_FRACTION", "0.75")
    d.prepare("mps")
    assert calls == [0.75]


def test_prepare_mps_keeps_default_when_unset(monkeypatch):
    calls = []
    monkeypatch.setattr(
        d.torch,
        "mps",
        SimpleNamespace(set_per_process_memory_fraction=lambda f: calls.append(f)),
    )
    monkeypatch.delenv("OXR_MPS_MEM_FRACTION", raising=False)
    d.prepare("mps")
    assert calls == []


def test_prepare_cpu_is_noop():
    d.prepare("cpu")


def test_move_model_keeps_cuda_where_device_map_placed_it(monkeypatch):
    class Model:
        def __init__(self):
            self.moves = []

        def to(self, dev):
            self.moves.append(dev)
            return "moved"

    model = Model()
    assert d.move_model(model, "cuda") is model
    assert model.moves == []


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_move_model_moves_other_devices(device):
    class Model:
        def __init__(self):
            self.moves = []

        def to(self, dev):
            self.moves.append(dev)
            return "moved"

    model = Model()
    assert d.move_model(model, device) == "moved"
    assert model.moves == [torch.device(device)]


def test_to_device_handles_none_tensor_tuple_and_dtype():
    assert d.to_device(None, torch.device("cpu")) is None

    x = torch.tensor([1, 2])
    assert d.to_device(x, torch.device("cpu"), torch.float16).dtype is torch.float16

    batched = d.to_device((torch.tensor([[1.0]]), torch.tensor([[2.0]])), torch.device("cpu"))
    assert batched.shape == (2, 1)


def test_memory_stats_on_cpu_are_zero():
    assert d.allocated_gb("cpu") == 0.0
    assert d.driver_gb("cpu") == 0.0
    assert d.peak_gb("cpu") == 0.0


def test_memory_stats_on_cuda(monkeypatch):
    monkeypatch.setattr(d.torch.cuda, "memory_allocated", lambda: 2 * 1024**3)
    monkeypatch.setattr(d.torch.cuda, "memory_reserved", lambda: 3 * 1024**3)
    monkeypatch.setattr(d.torch.cuda, "max_memory_allocated", lambda: 4 * 1024**3)
    assert d.allocated_gb("cuda") == 2.0
    assert d.driver_gb("cuda") == 3.0
    assert d.peak_gb("cuda") == 4.0


def test_memory_stats_on_mps(monkeypatch):
    monkeypatch.setattr(
        d.torch,
        "mps",
        SimpleNamespace(
            current_allocated_memory=lambda: 2 * 1024**3,
            driver_allocated_memory=lambda: 3 * 1024**3,
        ),
    )
    assert d.allocated_gb("mps") == 2.0
    assert d.driver_gb("mps") == 3.0
    # MPS 没有峰值 API，peak_gb 与 driver_gb 同口径
    assert d.peak_gb("mps") == 3.0


def test_empty_cache_dispatches(monkeypatch):
    calls = []
    monkeypatch.setattr(d.torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(d.torch, "mps", SimpleNamespace(empty_cache=lambda: calls.append("mps")))
    d.empty_cache("cuda")
    d.empty_cache("mps")
    d.empty_cache("cpu")
    assert calls == ["cuda", "mps"]


def test_peak_tracker_cuda_reads_official_counter(monkeypatch):
    reset = []
    monkeypatch.setattr(d.torch.cuda, "reset_peak_memory_stats", lambda: reset.append(1))
    monkeypatch.setattr(d.torch.cuda, "max_memory_allocated", lambda: 3 * 1024**3)

    with d.PeakTracker("cuda") as tracker:
        assert tracker.peak_gb == 0.0
    assert tracker.peak_gb == 3.0
    assert reset == [1]


def test_peak_tracker_mps_samples_driver_memory(monkeypatch):
    monkeypatch.setattr(
        d.torch, "mps", SimpleNamespace(driver_allocated_memory=lambda: 2 * 1024**3)
    )
    with d.PeakTracker("mps", interval=0.01) as tracker:
        time.sleep(0.03)
    assert tracker.peak_gb == 2.0
    assert tracker._stop.is_set()


def test_peak_tracker_cpu_is_zero():
    with d.PeakTracker("cpu") as tracker:
        pass
    assert tracker.peak_gb == 0.0


def test_describe_cpu_and_dtype():
    assert d.describe("cpu") == "CPU"
    assert d.describe("cpu", torch.float16) == "CPU, float16"


def test_describe_cuda(monkeypatch):
    monkeypatch.setattr(
        d.torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name="RTX 4090", total_memory=16 * 1024**3),
    )
    assert d.describe("cuda", torch.bfloat16) == "CUDA RTX 4090（总显存 16.0 GB, bfloat16）"


def test_describe_mps(monkeypatch):
    monkeypatch.setattr(
        d.torch,
        "mps",
        SimpleNamespace(recommended_max_memory=lambda: 64 * 1024**3),
    )
    assert d.describe("mps", torch.float16) == "Apple MPS（统一内存 64.0 GB, float16）"


def test_memory_line(monkeypatch):
    assert d.memory_line("cpu") == "CPU：无显存统计"
    monkeypatch.setattr(d, "driver_gb", lambda device: 1.5)
    monkeypatch.setattr(d, "allocated_gb", lambda device: 0.5)
    assert d.memory_line("mps") == "驱动占用 1.5 GB / 分配 0.5 GB"
