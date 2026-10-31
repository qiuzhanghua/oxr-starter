"""推理后端：本地 transformers（CUDA / MPS / CPU）或远端 vLLM HTTP 服务。

两者接口一致：recognize(image, prompt, max_tokens) -> str
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request

import torch
from PIL import Image

from . import devices

MODEL_ID = "SoMarkAI/OXR-1.0"

# 模型 preprocess 的默认上限（1344×1792）。
DEFAULT_MAX_PIXELS = 1344 * 1792
MIN_PIXELS = 448 * 448


def resolve_max_pixels(value: int | str | None = None) -> int | None:
    """解析 `--max-pixels` / `OXR_MAX_PIXELS`。

    0 / None / "auto" 表示用模型默认值（返回 None）；正整数优先于环境变量。
    视觉编码的注意力是 O(N²)，页面上限每降一半，视觉耗时约降到 1/4。
    """
    def _auto(v: object) -> bool:
        return v is None or v == "" or v == "auto" or v == 0 or v == "0"

    if _auto(value):
        value = os.environ.get("OXR_MAX_PIXELS")
        if _auto(value):
            return None
    try:
        pixels = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"max_pixels 必须是整数：{value!r}") from e
    if pixels < MIN_PIXELS:
        raise ValueError(
            f"max_pixels 太小（{pixels}），至少 {MIN_PIXELS}（448×448）"
        )
    return pixels


class LocalBackend:
    """在本进程内加载模型（transformers）。

    设备自适应：CUDA → MPS → CPU（见 `devices.pick_device`）。CUDA 仍用
    `bfloat16` + `device_map="cuda:0"`，与改造前完全一致；Apple Silicon 用
    `float16` + `.to("mps")`。
    """

    name = "local"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        dtype: str | None = None,
        max_pixels: int | str | None = None,
    ):
        from transformers import AutoModelForCausalLM

        self.device = devices.pick_device(device)
        self.dtype = devices.resolve_dtype(self.device, dtype)
        self.max_pixels = resolve_max_pixels(max_pixels)
        devices.prepare(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            **devices.device_kwargs(self.device),
        )
        self.model = devices.move_model(self.model, self.device)
        self.model.eval()

    def recognize(self, image: Image.Image, prompt: str, max_tokens: int = 2048) -> str:
        dev = devices.torch_device(self.device)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        preprocess_kwargs = (
            {} if self.max_pixels is None else {"max_pixels": self.max_pixels}
        )
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages, **preprocess_kwargs
        )
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids.to(dev),
                pixel_values=devices.to_device(pixel_values, dev),
                grid_thws=devices.to_device(grid_thws, dev),
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        if self.device == devices.MPS:
            torch.mps.synchronize()
        return self.model.text_tokenizer.decode(
            out[0], skip_special_tokens=True
        ).strip()

    def close(self) -> None:
        """释放模型与显存（服务里重启时有用）。"""
        self.model = None
        devices.empty_cache(self.device)


class VLLMBackend:
    """调用已启动的 vLLM（或本机 MPS 服务）的 OpenAI 兼容接口。"""

    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001/v1",
        model: str = "oxr",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _encode(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def recognize(self, image: Image.Image, prompt: str, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + self._encode(image)
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise SystemExit(f"vLLM 服务不可用（{self.base_url}）：{e}") from e
        return data["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        """HTTP 客户端无需释放（与 LocalBackend 接口对齐）。"""


def make_backend(
    kind: str = "local",
    base_url: str = "http://127.0.0.1:8001/v1",
    device: str | None = None,
    dtype: str | None = None,
    max_pixels: int | str | None = None,
):
    if kind == "vllm":
        return VLLMBackend(base_url)
    return LocalBackend(device=device, dtype=dtype, max_pixels=max_pixels)
