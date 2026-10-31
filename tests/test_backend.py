"""backend.py：本地 transformers 路径（用假模型）与 vLLM HTTP 路径（用桩服务）。"""

from __future__ import annotations

import base64
import io
import json
import socket
import sys
import types
from http.server import BaseHTTPRequestHandler

import pytest
import torch
from PIL import Image

from oxr_starter import backend, devices
from tests.helpers import serve_http


class FakeTokenizer:
    def __init__(self):
        self.seen: list[tuple] = []

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        self.seen.append((tokens, skip_special_tokens))
        return "  hello  "


class FakeModel:
    def __init__(self):
        self.text_tokenizer = FakeTokenizer()
        self.messages = None
        self.generate_kwargs = None
        self.moves = []
        self.evald = False

    def preprocess_inputs(self, messages, **kwargs):
        self.messages = messages
        self.preprocess_kwargs = kwargs
        return (
            torch.tensor([[1, 2, 3]]),
            torch.full((1, 2, 3), 0.5),
            torch.tensor([[1, 2]]),
        )

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[4, 5]])

    def to(self, dev):
        self.moves.append(dev)
        return self

    def eval(self):
        self.evald = True
        return self


@pytest.fixture
def fake_transformers(monkeypatch):
    """用假模型替换 AutoModelForCausalLM，记录 from_pretrained 的参数。"""
    holder = types.SimpleNamespace(kwargs=None, model=None)

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            holder.kwargs = {"model_id": model_id, **kwargs}
            holder.model = FakeModel()
            return holder.model

    monkeypatch.delenv("OXR_MAX_PIXELS", raising=False)
    module = types.ModuleType("transformers")
    module.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    return holder


# --------------------------------------------------------------------------- #
# LocalBackend：设备放置
# --------------------------------------------------------------------------- #
def test_local_backend_cpu(fake_transformers):
    b = backend.LocalBackend(model_id="my/model", device="cpu", dtype="fp32")

    assert b.device == "cpu"
    assert b.dtype is torch.float32
    assert fake_transformers.kwargs["model_id"] == "my/model"
    assert fake_transformers.kwargs["trust_remote_code"] is True
    assert fake_transformers.kwargs["low_cpu_mem_usage"] is True
    assert "device_map" not in fake_transformers.kwargs
    assert fake_transformers.model.evald is True
    assert fake_transformers.model.moves == [torch.device("cpu")]


def test_local_backend_cuda_keeps_device_map(fake_transformers, monkeypatch):
    monkeypatch.setattr(devices, "pick_device", lambda prefer=None: "cuda")
    monkeypatch.setattr(devices, "prepare", lambda device: None)

    b = backend.LocalBackend(model_id="my/model")

    assert b.device == "cuda"
    assert b.dtype is torch.bfloat16
    assert fake_transformers.kwargs["device_map"] == "cuda:0"
    assert fake_transformers.kwargs["dtype"] is torch.bfloat16
    # CUDA 由 device_map 放置，不再显式 .to()
    assert fake_transformers.model.moves == []


def test_local_backend_mps_uses_fp16_and_moves_model(fake_transformers, monkeypatch):
    monkeypatch.setattr(devices, "pick_device", lambda prefer=None: "mps")
    monkeypatch.setattr(devices, "prepare", lambda device: None)

    b = backend.LocalBackend(model_id="my/model")

    assert b.device == "mps"
    assert b.dtype is torch.float16
    assert "device_map" not in fake_transformers.kwargs
    assert fake_transformers.kwargs["dtype"] is torch.float16
    assert fake_transformers.model.moves == [torch.device("mps")]


# --------------------------------------------------------------------------- #
# LocalBackend.recognize
# --------------------------------------------------------------------------- #
def test_local_backend_recognize_builds_messages_and_moves_inputs(fake_transformers):
    b = backend.LocalBackend(model_id="my/model", device="cpu", dtype="fp32")
    image = Image.new("RGB", (8, 6), "white")

    out = b.recognize(image, "describe", max_tokens=9)

    model = fake_transformers.model
    assert out == "hello"  # 两端空白被 strip
    content = model.messages[0]["content"]
    assert content[0] == {"type": "image", "image": image}
    assert content[1] == {"type": "text", "text": "describe"}
    assert model.preprocess_kwargs == {}

    kwargs = model.generate_kwargs
    assert kwargs["max_new_tokens"] == 9
    assert kwargs["do_sample"] is False
    assert kwargs["inputs"].device.type == "cpu"
    assert kwargs["pixel_values"].shape == (1, 2, 3)
    assert kwargs["grid_thws"].tolist() == [[1, 2]]

    tokens, skip = model.text_tokenizer.seen[0]
    assert tokens.tolist() == [4, 5]
    assert skip is True


def test_local_backend_recognize_synchronizes_mps(fake_transformers, monkeypatch):
    b = backend.LocalBackend(model_id="my/model", device="cpu", dtype="fp32")
    b.device = "mps"
    synced = []
    monkeypatch.setattr(devices, "torch_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synced.append(1))
    monkeypatch.setattr(devices, "empty_cache", lambda device: None)

    b.recognize(Image.new("RGB", (4, 4)), "prompt", 5)

    assert synced == [1]


def test_local_backend_passes_max_pixels(fake_transformers):
    b = backend.LocalBackend(
        model_id="my/model", device="cpu", dtype="fp32", max_pixels=1_200_000
    )

    b.recognize(Image.new("RGB", (8, 6)), "describe", max_tokens=9)

    assert b.max_pixels == 1_200_000
    assert fake_transformers.model.preprocess_kwargs == {"max_pixels": 1_200_000}


def test_local_backend_reads_max_pixels_from_env(fake_transformers, monkeypatch):
    monkeypatch.setenv("OXR_MAX_PIXELS", "900000")

    b = backend.LocalBackend(model_id="my/model", device="cpu", dtype="fp32")

    assert b.max_pixels == 900_000
    b.recognize(Image.new("RGB", (8, 6)), "describe", max_tokens=9)
    assert fake_transformers.model.preprocess_kwargs == {"max_pixels": 900_000}


def test_explicit_max_pixels_overrides_env(fake_transformers, monkeypatch):
    monkeypatch.setenv("OXR_MAX_PIXELS", "900000")

    b = backend.LocalBackend(
        model_id="my/model", device="cpu", dtype="fp32", max_pixels=1_500_000
    )

    assert b.max_pixels == 1_500_000


@pytest.mark.parametrize(
    ("value", "env", "expected"),
    [
        (None, None, None),
        (0, None, None),
        ("auto", None, None),
        ("", "1200000", 1_200_000),
        (None, "1200000", 1_200_000),
        (None, "0", None),
        (900000, "1200000", 900_000),
    ],
)
def test_resolve_max_pixels(value, env, expected, monkeypatch):
    if env is None:
        monkeypatch.delenv("OXR_MAX_PIXELS", raising=False)
    else:
        monkeypatch.setenv("OXR_MAX_PIXELS", env)
    assert backend.resolve_max_pixels(value) == expected


@pytest.mark.parametrize("value", ["abc", "-1", 1000, 100])
def test_resolve_max_pixels_rejects_bad_values(value, monkeypatch):
    monkeypatch.delenv("OXR_MAX_PIXELS", raising=False)
    with pytest.raises(ValueError):
        backend.resolve_max_pixels(value)


def test_local_backend_close_releases_model_and_cache(fake_transformers, monkeypatch):
    emptied = []
    monkeypatch.setattr(devices, "empty_cache", lambda device: emptied.append(device))
    b = backend.LocalBackend(model_id="my/model", device="cpu", dtype="fp32")

    b.close()

    assert b.model is None
    assert emptied == ["cpu"]


# --------------------------------------------------------------------------- #
# VLLMBackend
# --------------------------------------------------------------------------- #
def test_vllm_encode_produces_png():
    encoded = backend.VLLMBackend()._encode(Image.new("RGB", (7, 5), "blue"))
    raw = base64.b64decode(encoded)

    with Image.open(io.BytesIO(raw)) as img:
        assert img.format == "PNG"
        assert img.size == (7, 5)


def test_vllm_recognize_posts_openai_payload():
    captured: dict = {}

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            captured["path"] = self.path
            captured["payload"] = json.loads(self.rfile.read(length))
            body = json.dumps({"choices": [{"message": {"content": "  markdown  "}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    with serve_http(Stub) as base:
        client = backend.VLLMBackend(base_url=base + "/v1/", model="m")
        out = client.recognize(Image.new("RGB", (120, 80), "red"), "describe", 77)

    assert out == "markdown"
    assert captured["path"] == "/v1/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == "m"
    assert payload["max_tokens"] == 77
    assert payload["temperature"] == 0
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "describe"}

    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
        assert img.size == (120, 80)


def test_vllm_recognize_maps_http_error_to_system_exit():
    class Failing(BaseHTTPRequestHandler):
        def do_POST(self):
            body = b"{}"
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    with serve_http(Failing) as base, pytest.raises(SystemExit, match="vLLM 服务不可用"):
        backend.VLLMBackend(base_url=base + "/v1", timeout=2).recognize(
            Image.new("RGB", (4, 4)), "p", 4
        )


def test_vllm_recognize_maps_connection_error_to_system_exit():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    client = backend.VLLMBackend(base_url=f"http://127.0.0.1:{port}/v1", timeout=1)
    try:
        with pytest.raises(SystemExit, match="vLLM 服务不可用"):
            client.recognize(Image.new("RGB", (4, 4)), "p", 4)
    finally:
        sock.close()


def test_vllm_close_is_noop():
    backend.VLLMBackend().close()


# --------------------------------------------------------------------------- #
# make_backend
# --------------------------------------------------------------------------- #
def test_make_backend_returns_vllm_client():
    client = backend.make_backend("vllm", base_url="http://h:9/v1/")
    assert isinstance(client, backend.VLLMBackend)
    assert client.base_url == "http://h:9/v1"
    assert client.name == "vllm"


def test_make_backend_passes_device_options_to_local(monkeypatch):
    created = {}

    class FakeLocal:
        name = "local"

        def __init__(self, **kwargs):
            created.update(kwargs)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(backend, "LocalBackend", FakeLocal)

    client = backend.make_backend("local", device="mps", dtype="fp16", max_pixels=1_200_000)

    assert isinstance(client, FakeLocal)
    assert created["device"] == "mps"
    assert created["dtype"] == "fp16"
    assert created["max_pixels"] == 1_200_000
