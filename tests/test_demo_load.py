"""generate_demo / load_model 的 CLI 路径（假模型，不下载权重）。"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from PIL import Image

from oxr_starter import devices, generate_demo, load_model


class FakeTokenizer:
    def __init__(self, size: int = 100):
        self.size = size
        self.seen: list[torch.Tensor] = []

    def __len__(self) -> int:
        return self.size

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        self.seen.append(tokens)
        return "decoded markdown"


class FakeModel:
    def __init__(self, num_params: int = 0):
        self.text_tokenizer = FakeTokenizer()
        self.messages = None
        self.generate_kwargs = None
        self.moves = []
        self.evald = False
        self._params = [torch.nn.Parameter(torch.zeros(num_params))] if num_params else []

    def preprocess_inputs(self, messages, **kwargs):
        self.messages = messages
        self.preprocess_kwargs = kwargs
        return (
            torch.tensor([[1, 2, 3]]),
            torch.zeros(1, 2, 3),
            torch.tensor([[1, 2]]),
        )

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[10, 11, 12]])

    def parameters(self):
        return iter(self._params)

    def to(self, dev):
        self.moves.append(dev)
        return self

    def eval(self):
        self.evald = True
        return self


@pytest.fixture
def fake_transformers(monkeypatch):
    """让 generate_demo / load_model 拿到假模型；强制走 CPU 分支。"""
    holder = types.SimpleNamespace(model=None, kwargs=None, tokenizer=None)

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            holder.kwargs = {"model_id": model_id, **kwargs}
            holder.model = FakeModel(num_params=2_000_000)
            return holder.model

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            holder.tokenizer = FakeTokenizer()
            return holder.tokenizer

    monkeypatch.delenv("OXR_MAX_PIXELS", raising=False)
    module = types.ModuleType("transformers")
    module.AutoModelForCausalLM = FakeAutoModel
    module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(devices, "pick_device", lambda prefer=None: "cpu")
    return holder


def test_generate_demo_uses_builtin_sample(monkeypatch, fake_transformers, capsys):
    monkeypatch.setattr(sys, "argv", ["oxr-demo"])

    generate_demo.main()

    out = capsys.readouterr().out
    assert "输入图: (900, 620)" in out
    assert "生成：3 tokens" in out
    assert "decoded markdown" in out
    assert "显存: 无（CPU）" in out

    model = fake_transformers.model
    assert model.evald is True
    assert model.moves == [torch.device("cpu")]
    assert model.generate_kwargs["max_new_tokens"] == 512
    assert model.generate_kwargs["do_sample"] is False

    content = model.messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["image"].size == (900, 620)
    assert content[1] == {"type": "text", "text": generate_demo.PROMPT_WHOLE_PAGE}
    assert model.preprocess_kwargs == {}


def test_generate_demo_reads_image_and_max_tokens(
    monkeypatch, fake_transformers, tmp_path, capsys
):
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 8), "white").save(image)
    monkeypatch.setattr(sys, "argv", ["oxr-demo", str(image), "--max-tokens", "7"])

    generate_demo.main()

    assert "输入图: (10, 8)" in capsys.readouterr().out
    assert fake_transformers.model.generate_kwargs["max_new_tokens"] == 7


def test_generate_demo_passes_max_pixels(monkeypatch, fake_transformers, capsys):
    monkeypatch.setattr(sys, "argv", ["oxr-demo", "--max-pixels", "900000"])

    generate_demo.main()

    assert fake_transformers.model.preprocess_kwargs == {"max_pixels": 900000}
    assert "max_pixels 900000" in capsys.readouterr().out


def test_generate_demo_passes_device_kwargs(monkeypatch, fake_transformers):
    monkeypatch.setattr(sys, "argv", ["oxr-demo"])
    generate_demo.main()

    kwargs = fake_transformers.kwargs
    assert kwargs["model_id"] == generate_demo.MODEL_ID
    assert kwargs["dtype"] is torch.float32
    assert kwargs["trust_remote_code"] is True
    assert "device_map" not in kwargs


def test_load_model_reports_params_and_memory(monkeypatch, fake_transformers, capsys):
    monkeypatch.setattr(sys, "argv", ["oxr-load"])

    load_model.main()

    out = capsys.readouterr().out
    assert "CPU, float32" in out
    assert "vocab 100" in out
    assert "参数量: 0.002 B" in out
    assert "峰值显存: 0.0 GB" in out
    assert "CPU：无显存统计" in out
    assert "OK: 模型可以加载" in out

    model = fake_transformers.model
    assert model.evald is True
    assert model.moves == [torch.device("cpu")]
